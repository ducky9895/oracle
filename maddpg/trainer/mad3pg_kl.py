import numpy as np
import tensorflow as tf
from gym import Space
from gym.spaces import Discrete

from maddpg.trainer.AbstractAgent import AbstractAgent
from maddpg.trainer.maddpg_kl import MADDPGCriticNetwork, MADDPGklPolicyNetwork
from maddpg.common.util import space_n_to_shape_n, clip_by_local_norm

class MAD3PGklAgent(AbstractAgent):
    def __init__(self, obs_space_n, act_space_n, agent_index, batch_size, buff_size, lr, num_layer, num_units, gamma,
                 tau, prioritized_replay=False, alpha=0.6, max_step=None, initial_beta=0.6, prioritized_replay_eps=1e-6,
                 _run=None, num_atoms=51, min_val=-150, max_val=0):
        """
        Implementation of a Multi-Agent version of D3PG (Distributed Deep Deterministic Policy
        Gradient).

        num_atoms, min_val and max_val control the parametrization of the value function.
        """
        self._run = _run

        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = space_n_to_shape_n(obs_space_n)
        act_shape_n = space_n_to_shape_n(act_space_n)
        super().__init__(buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay, alpha, max_step, initial_beta,
                         prioritized_replay_eps=prioritized_replay_eps)

        act_type = type(act_space_n[0])
        self.critic = CatDistCritic(2, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index, num_atoms, min_val, max_val)
        self.critic_target = CatDistCritic(2, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index, num_atoms, min_val, max_val)
        self.critic_target.model.set_weights(self.critic.model.get_weights())

        self.local_policy = MADDPGklPolicyNetwork(2, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                                  self.critic, agent_index)
        self.full_policy = MADDPGklPolicyNetwork(2, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                                 self.critic, agent_index, full=True)
        
        self.local_policy_target = MADDPGklPolicyNetwork(2, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                                         self.critic, agent_index)
        self.full_policy_target = MADDPGklPolicyNetwork(2, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                                        self.critic, agent_index, full=True)

        self.local_policy_target.model.set_weights(self.local_policy.model.get_weights())
        self.full_policy_target.model.set_weights(self.full_policy.model.get_weights())

        self.batch_size = batch_size
        self.agent_index = agent_index
        self.gamma = gamma
        self.tau = tau

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

    def update_target_networks(self):
        def update_target_network(net, target_net, tau):
            net_weights = np.array(net.get_weights())
            target_net_weights = np.array(target_net.get_weights())
            new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            target_net.set_weights(new_weights)

        update_target_network(self.critic.model, self.critic_target.model, self.tau)
        update_target_network(self.local_policy.model, self.local_policy_target.model, self.tau)
        update_target_network(self.full_policy.model, self.full_policy_target.model, self.tau)

    def update(self, agents, step):
        assert agents[self.agent_index] is self

        if self.prioritized_replay:
            obs_n, acts_n, rew_n, next_obs_n, done_n, weights, indices = \
                self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step))
        else:
            obs_n, acts_n, rew_n, next_obs_n, done_n = self.replay_buffer.sample(self.batch_size)
            weights = tf.ones(rew_n.shape)
 
        target_act_next = []
        for a, obs in zip(agents, next_obs_n):
            if a is self:
                target_act_next.append(a.target_action_full(next_obs_n))
            else:
                target_act_next.append(a.target_action(obs))

        target_prob_next = self.critic_target.predict_probs(next_obs_n, target_act_next)
        q_next = np.sum(target_prob_next * self.critic.atoms, 1)  # note: maybe change this to tf to speed_up

        atoms_next = rew_n[:, None] + self.gamma * self.critic.atoms
        atoms_next = np.clip(atoms_next, self.critic.min_val, self.critic.max_val).astype(np.float32)

        target_prob = self.project_distribution(atoms_next, target_prob_next)

        # apply update
        td_loss = self.critic.train_step(obs_n, acts_n, target_prob, weights).numpy()

        # Compute KL divergence between local policy and target full policy for the given agent_index
        local_act_probs = self.local_policy.model(next_obs_n[self.agent_index][None])
        target_act_probs = self.full_policy_target.model(np.concatenate(next_obs_n, axis=-1)[None])
        KL_loss = tf.reduce_mean(tf.keras.losses.KLD(target_act_probs, local_act_probs))

        local_policy_loss = self.local_policy.train(obs_n, acts_n) + KL_loss * 0.01
        full_policy_loss = self.full_policy.train(obs_n, acts_n)

        if self.prioritized_replay:
            self.replay_buffer.update_priorities(indices, td_loss + self.prioritized_replay_eps)

        self.update_target_networks()

        self._run.log_scalar(f'agent_{self.agent_index}.train.local_policy_loss', local_policy_loss.numpy(), step)
        self._run.log_scalar(f'agent_{self.agent_index}.train.full_policy_loss', full_policy_loss.numpy(), step)
        self._run.log_scalar(f'agent_{self.agent_index}.train.q_loss0', np.mean(td_loss), step)

        return [td_loss, local_policy_loss]

    def project_distribution(self, atoms_next, target_prob_next):
        """
        Projects the distribution onto the new support.
        Includes in the comments a non-vectorized version, which is a lot slower, although the new one is still a bit
        slow.
        TODO: this numpy only solution is a bit slow.
        """
        b = (atoms_next - self.critic.min_val) / self.critic.delta_atom  # b = continuous 'index' of atom being projected to
        lower = np.floor(b)
        upper = np.ceil(b)
        # determine/interpolate membership to different atoms
        density_lower = target_prob_next * (upper + np.float32(
            lower == upper) - b)  # note: not sure about lower == upper, that's stolen from ShangtongZhang
        density_upper = target_prob_next * (b - lower)
        # sum up membergship from all projected atoms to target probability
        target_prob = np.zeros(target_prob_next.shape, dtype=np.float32)
        for batch_idx in range(self.batch_size):  # todo it would be better to vectorize per atom, not per batch idx...
            target_prob[batch_idx, np.int32(lower[batch_idx, :])] += density_lower[batch_idx, :]
            target_prob[batch_idx, np.int32(upper[batch_idx, :])] += density_upper[batch_idx, :]
        return target_prob

    def save(self, fp):
        self.critic.model.save_weights(fp + 'critic.h5',)
        self.critic_target.model.save_weights(fp + 'critic_target.h5')
        self.local_policy.model.save_weights(fp + 'local_policy.h5')
        self.local_policy_target.model.save_weights(fp + 'local_policy_target.h5')
        self.full_policy.model.save_weights(fp + 'full_policy.h5')
        self.full_policy_target.model.save_weights(fp + 'full_policy_target.h5')

    def load(self, fp):
        self.critic.model.load_weights(fp + 'critic.h5')
        self.critic_target.model.load_weights(fp + 'critic_target.h5')
        self.local_policy.model.load_weights(fp + 'local_policy.h5')
        self.local_policy_target.model.load_weights(fp + 'local_policy_target.h5')
        self.full_policy.model.load_weights(fp + 'full_policy.h5')
        self.full_policy_target.model.load_weights(fp + 'full_policy_target.h5')

class CatDistCritic(MADDPGCriticNetwork):
    def __init__(self, num_hidden_layers, units_per_layer, lr, obs_n_shape, act_shape_n, act_type, agent_index,
                 n_atoms, min_val, max_val):
        """
        Implementation of a critic that outputs a categorical distribution, similar to how it was used in both
        D4PG and the original Bellemare Distributional paper.
        regression ANN.
        """
        super().__init__(num_hidden_layers, units_per_layer, lr, obs_n_shape, act_shape_n,
                         act_type, agent_index)
        self.n_atoms = n_atoms
        self.atoms = np.linspace(min_val, max_val, n_atoms)
        self.delta_atom = (max_val - min_val) / (n_atoms - 1)
        self.min_val = min_val
        self.max_val = max_val

        # replace output layer from normal critic
        self.output_layer = tf.keras.layers.Dense(self.n_atoms, activation='softmax',
                                                  name='ag{}crit_out'.format(agent_index))

        # connect layers
        x = self.input_concat_layer(self.obs_input_n + self.act_input_n)
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=self.obs_input_n + self.act_input_n,  # list concatenation
                                    outputs=[x])
        self.model.compile(self.optimizer, loss='mse')

    def predict_probs(self, obs_n, act_n):
        """
        Return the probabilities for the given input.
        """
        return self._predict_internal_probs(obs_n + act_n)

    def predict_expectation(self, obs_n, act_n):
        """
        Predict the expectation of the distribution for given input.
        """
        return self._predict_internal(obs_n + act_n)

    @tf.function
    def _predict_internal(self, concatenated_input):
        """
        Returns the expected value, i.e. sums up the probs * values.
        """
        probs = self._predict_internal_probs(concatenated_input)
        dist = probs * self.atoms
        return tf.math.reduce_sum(dist, 1)

    @tf.function
    def _predict_internal_probs(self, concatenated_input):
        """
        Returns the probabilities for a given batch of inputs.
        """
        x = self.input_concat_layer(concatenated_input)
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)
        # x = tf.math.softmax(x)
        return x

    def train_step(self, obs_n, act_n, target_prob, weights):
        """
        Train the critic network with the observations, actions, rewards and next observations,
        and next actions.
        """
        return self._train_step_internal(obs_n + act_n, target_prob, weights)

    @tf.function
    def _train_step_internal(self, concatenated_input, target_prob, weights):
        """
        Internal function, because concatenation can not be done inside tf.function.
        """
        with tf.GradientTape(persistent=True) as tape:
            x = self.input_concat_layer(concatenated_input)
            for idx in range(self.num_layers):
                x = self.hidden_layers[idx](x)
            x = self.output_layer(x)
            q_pred = x

            crossent_loss = tf.losses.binary_crossentropy(target_prob, q_pred)
            loss = crossent_loss

        gradients = tape.gradient(loss, self.model.trainable_variables)

        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return crossent_loss
