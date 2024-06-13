import numpy as np
import tensorflow as tf

from gym import Space

from maddpg.trainer.AbstractAgent import AbstractAgent
from maddpg.trainer.maddpg import MADDPGCriticNetwork, MADDPGPolicyNetwork
from maddpg.common.util import space_n_to_shape_n, clip_by_local_norm


class MASACklAgent(AbstractAgent):
    def __init__(self, obs_space_n, act_space_n, agent_index, batch_size, buff_size, lr, num_layer, num_units, gamma,
                 tau, prioritized_replay=False, alpha=0.6, max_step=None, initial_beta=0.6, prioritized_replay_eps=1e-6,
                 entropy_coeff=0.2, use_gauss_policy=False, use_gumbel=True, policy_update_freq=1, _run=None,
                 multi_step=1):
        """
        Implementation of Multi-Agent Soft-Actor-Critic with KL divergence and delayed policy updates.
        """
        self._run = _run
        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = space_n_to_shape_n(obs_space_n)
        act_shape_n = space_n_to_shape_n(act_space_n)
        super().__init__(buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay, alpha, max_step, initial_beta,
                         prioritized_replay_eps=prioritized_replay_eps)

        act_type = type(act_space_n[0])
        self.critic_1 = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_1_target = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_1_target.model.set_weights(self.critic_1.model.get_weights())
        self.critic_2 = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_2_target = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_2_target.model.set_weights(self.critic_2.model.get_weights())

        self.local_policy = MASACklPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1, entropy_coeff,
                                                 agent_index, self.critic_1, use_gauss_policy, use_gumbel, prioritized_replay_eps)
        self.full_policy = MASACklPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1, entropy_coeff,
                                                agent_index, self.critic_1, use_gauss_policy, use_gumbel, prioritized_replay_eps, full=True)
        
        self.local_policy_target = MASACklPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1, entropy_coeff,
                                                        agent_index, self.critic_1, use_gauss_policy, use_gumbel, prioritized_replay_eps)
        self.full_policy_target = MASACklPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1, entropy_coeff,
                                                       agent_index, self.critic_1, use_gauss_policy, use_gumbel, prioritized_replay_eps, full=True)

        self.local_policy_target.model.set_weights(self.local_policy.model.get_weights())
        self.full_policy_target.model.set_weights(self.full_policy.model.get_weights())

        self.use_gauss_policy = use_gauss_policy
        self.use_gumbel = use_gumbel
        self.policy_update_freq = policy_update_freq

        self.batch_size = batch_size
        self.decay = gamma
        self.tau = tau
        self.entropy_coeff = entropy_coeff
        self.update_counter = 0
        self.agent_index = agent_index
        self.multi_step = multi_step

    def action(self, obs):
        return self.local_policy.get_action(obs[None])[0]

    def action_full(self, obs):
        return self.full_policy.get_action(np.concatenate(obs, axis=-1)[None])[0]

    def target_action(self, obs):
        return self.local_policy_target.get_action(obs)

    def target_action_full(self, obs):
        return self.full_policy_target.get_action(np.concatenate(obs, axis=-1))

    def preupdate(self):
        pass

    def update_target_networks(self, tau):
        def update_target_network(net: tf.keras.Model, target_net: tf.keras.Model):
            net_weights = np.array(net.get_weights())
            target_net_weights = np.array(target_net.get_weights())
            new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            target_net.set_weights(new_weights)

        update_target_network(self.critic_1.model, self.critic_1_target.model)
        update_target_network(self.critic_2.model, self.critic_2_target.model)
        update_target_network(self.local_policy.model, self.local_policy_target.model)
        update_target_network(self.full_policy.model, self.full_policy_target.model)

    def update(self, agents, step):
        assert agents[self.agent_index] is self
        self.update_counter += 1

        if self.prioritized_replay:
            obs_n, acts_n, rew_n, next_obs_n, done_n, weights, indices = \
                self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step))
        else:
            obs_n, acts_n, rew_n, next_obs_n, done_n = self.replay_buffer.sample(self.batch_size)
            weights = tf.ones(rew_n.shape)

        next_act_sampled_n = [ag.target_action(next_obs) for ag, next_obs in zip(agents, next_obs_n)]
        if self.use_gauss_policy:
            logact_probs = self.local_policy.action_logprob(next_obs_n[self.agent_index], next_act_sampled_n[self.agent_index])[:, None]  # only our own entropy is 'controllable'
            entropy = -logact_probs
        elif self.use_gumbel:
            action_probs = self.local_policy.get_all_action_probs(next_obs_n[self.agent_index])
            action_log_probs = np.log(action_probs + self.prioritized_replay_eps)
            buff = -action_probs * action_log_probs
            entropy = np.sum(buff, 1)

        critic_outputs = np.empty([2, self.batch_size], dtype=np.float32)  # this is a lot faster than python list plus minimum
        critic_outputs[0] = self.critic_1_target.predict(next_obs_n, next_act_sampled_n)[:, 0]
        critic_outputs[1] = self.critic_2_target.predict(next_obs_n, next_act_sampled_n)[:, 0]
        q_min = np.min(critic_outputs, 0)[:, None]

        target_q = rew_n[:, None] + self.decay * (q_min + self.entropy_coeff * entropy)

        td_loss = np.empty([2, self.batch_size], dtype=np.float32)
        td_loss[0] = self.critic_1.train_step(obs_n, acts_n, target_q, weights).numpy()[:, 0]
        td_loss[1] = self.critic_2.train_step(obs_n, acts_n, target_q, weights).numpy()[:, 0]
        td_loss_max = np.max(td_loss, 0)

        # Compute KL divergence between local policy and target full policy for the given agent_index
        local_act_probs = self.local_policy.model(next_obs_n[self.agent_index][None])
        target_act_probs = self.full_policy_target.model(np.concatenate(next_obs_n, axis=-1)[None])
        KL_loss = tf.reduce_mean(tf.keras.losses.KLD(target_act_probs, local_act_probs))

        local_policy_loss = self.local_policy.train(obs_n, acts_n) + KL_loss * 0.01
        full_policy_loss = self.full_policy.train(obs_n, acts_n)

        if self.prioritized_replay:
            self.replay_buffer.update_priorities(indices, td_loss_max + self.prioritized_replay_eps)

        if self.update_counter % self.policy_update_freq == 0:  # delayed policy updates
            policy_loss = local_policy_loss + full_policy_loss
            self.update_target_networks(self.tau)
            self._run.log_scalar(f'agent_{self.agent_index}.train.policy_loss', policy_loss.numpy(), step)
        else:
            policy_loss = None

        self._run.log_scalar(f'agent_{self.agent_index}.train.q_loss0', np.mean(td_loss[0]), step)
        self._run.log_scalar(f'agent_{self.agent_index}.train.q_loss1', np.mean(td_loss[1]), step)
        self._run.log_scalar(f'agent_{self.agent_index}.train.entropy', np.mean(entropy), step)

        return [td_loss, policy_loss]

    def save(self, fp):
        self.critic_1.model.save_weights(fp + 'critic_1.h5',)
        self.critic_2.model.save_weights(fp + 'critic_2.h5')
        self.critic_1_target.model.save_weights(fp + 'critic_target_1.h5',)
        self.critic_2_target.model.save_weights(fp + 'critic_target_2.h5')

        self.local_policy.model.save_weights(fp + 'local_policy.h5')
        self.local_policy_target.model.save_weights(fp + 'local_policy_target.h5')
        self.full_policy.model.save_weights(fp + 'full_policy.h5')
        self.full_policy_target.model.save_weights(fp + 'full_policy_target.h5')


    def load(self, fp):
        self.critic_1.model.load_weights(fp + 'critic_1.h5')
        self.critic_2.model.load_weights(fp + 'critic_2.h5')
        self.critic_1_target.model.load_weights(fp + 'critic_target_1.h5',)
        self.critic_2_target.model.load_weights(fp + 'critic_target_2.h5')

        self.local_policy.model.load_weights(fp + 'local_policy.h5')
        self.local_policy_target.model.load_weights(fp + 'local_policy_target.h5')
        self.full_policy.model.load_weights(fp + 'full_policy.h5')
        self.full_policy_target.model.load_weights(fp + 'full_policy_target.h5')


class MASACklPolicyNetwork(MADDPGPolicyNetwork):
    def __init__(self, num_layers, units_per_layer, lr, obs_n_shape, act_shape, act_type,
                 gumbel_temperature, entropy_coeff, agent_index, q_network, use_gaussian, use_gumbel,
                 numeric_eps, full=False):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final
        layer. Currently only implemented for discrete spaces with a gumbel policy.
        """
        super().__init__(num_layers, units_per_layer, lr, obs_n_shape, act_shape, act_type,
                         gumbel_temperature, q_network, agent_index)
        self.full = full
        self.use_gaussian = use_gaussian
        self.use_gumbel = use_gumbel
        self.entropy_coeff = entropy_coeff
        self.numeric_eps = numeric_eps

        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        ### set up network structure
        self.obs_input = tf.keras.layers.Input(shape=np.sum(self.obs_n_shape, axis=0)) if self.full else tf.keras.layers.Input(shape=self.obs_n_shape[agent_index])

        self.hidden_layers = []
        for idx in range(num_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}pol_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        if self.use_gumbel:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='linear',
                                                      name='ag{}pol_out{}'.format(agent_index, idx))
        else:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='tanh',
                                                      name='ag{}pol_out{}'.format(agent_index, idx))

        # connect layers
        x = self.obs_input
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=[self.obs_input], outputs=[x])

    @tf.function
    def get_all_action_probs(self, obs):
        logits = self.forward_pass(obs)
        return tf.math.softmax(logits)

    @tf.function
    def action_logprob(self, obs, action):
        logits = self.forward_pass(obs)

    @tf.function
    def train(self, obs_n, act_n):
        with tf.GradientTape() as tape:
            if self.full:
                p_input = tf.concat(obs_n, axis=-1)
            else:
                p_input = obs_n[self.agent_index]

            x = self.forward_pass(p_input)
            act_n = [a for a in act_n]
            if self.use_gumbel:
                logits = x
                act_n[self.agent_index] = self.gumbel_softmax_sample(logits)
                act_probs = tf.math.softmax(logits)
                entropy = - tf.math.reduce_sum(act_probs * tf.math.log(act_probs + self.numeric_eps), 1)
            elif self.use_gaussian:
                logits = x
                act_n[self.agent_index] = self.gaussian_sample(logits)
                entropy = - self.action_logprob(obs_n[self.agent_index], act_n[self.agent_index])
            q_value = self.q_network._predict_internal(obs_n + act_n)

            loss = -tf.math.reduce_mean(q_value + self.entropy_coeff * entropy)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return loss
