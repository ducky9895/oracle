import numpy as np
import tensorflow as tf

from gym import Space
from gym.spaces import Discrete

from maddpg.trainer.AbstractAgent import AbstractAgent
from maddpg.common.util import space_n_to_shape_n, clip_by_local_norm

class MADDPGklAgent(AbstractAgent):
    def __init__(self, obs_space_n, act_space_n, agent_index, batch_size, buff_size, lr, num_layer, num_units, gamma,
                 tau, prioritized_replay=False, alpha=0.6, max_step=None, initial_beta=0.6, prioritized_replay_eps=1e-6,
                 _run=None):
        self._run = _run

        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = space_n_to_shape_n(obs_space_n)
        act_shape_n = space_n_to_shape_n(act_space_n)
        super().__init__(buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay, alpha, max_step, initial_beta,
                         prioritized_replay_eps=prioritized_replay_eps)

        act_type = type(act_space_n[0])
        self.critic = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_target = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_target.model.set_weights(self.critic.model.get_weights())

        self.local_policy = MADDPGklPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                                  self.critic, agent_index)
        self.full_policy = MADDPGklPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                                 self.critic, agent_index, full=True)
        
        self.local_policy_target = MADDPGklPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                                         self.critic, agent_index)
        self.full_policy_target = MADDPGklPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
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

        target_q_next = self.critic_target.predict(next_obs_n, target_act_next)
        q_train_target = rew_n[:, None] + self.gamma * target_q_next * (1 - done_n[:, None])

        td_loss = self.critic.train_step(obs_n, acts_n, q_train_target, weights).numpy()[:, 0]

        # local_act_next = [a.action(obs) for a, obs in zip(agents, next_obs_n)]

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

    def save(self, fp):
        self.critic.model.save_weights(fp + 'critic.h5')
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



class MADDPGklPolicyNetwork(object):
    def __init__(self, num_layers, units_per_layer, lr, obs_n_shape, act_shape, act_type,
                 gumbel_temperature, q_network, agent_index, full=False):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final layer.
        """
        self.full=full
        self.num_layers = num_layers
        self.lr = lr
        self.obs_n_shape = obs_n_shape
        self.act_shape = act_shape
        self.act_type = act_type
        if act_type is Discrete:
            self.use_gumbel = True
        else:
            self.use_gumbel = False
        self.gumbel_temperature = gumbel_temperature
        self.q_network = q_network
        self.agent_index = agent_index
        self.clip_norm = 0.5

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

    @classmethod
    def gumbel_softmax_sample(cls, logits):
        """
        Produces Gumbel softmax samples from the input log-probabilities (logits).
        These are used, because they are differentiable approximations of the distribution of an argmax.
        """
        uniform_noise = tf.random.uniform(tf.shape(logits))
        gumbel = -tf.math.log(-tf.math.log(uniform_noise))
        noisy_logits = gumbel + logits  # / temperature
        return tf.math.softmax(noisy_logits)

    def forward_pass(self, obs):
        """
        Performs a simple forward pass through the NN.
        """
        x = obs
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        outputs = self.output_layer(x)  # log probabilities of the gumbel softmax dist are the output of the network
        return outputs

    @tf.function(experimental_relax_shapes=True)
    def get_action(self, obs):
        outputs = self.forward_pass(obs)
        if self.use_gumbel:
            outputs = self.gumbel_softmax_sample(outputs)
        return outputs

    @tf.function(experimental_relax_shapes=True)
    def train(self, obs_n, act_n):
        with tf.GradientTape() as tape:
            if self.full:
                p_input = tf.concat(obs_n, axis=-1)
            else:
                p_input = obs_n[self.agent_index]
            
            x = self.forward_pass(p_input)
            act_n = [a for a in act_n]
            if self.use_gumbel:
                logits = x  # log probabilities of the gumbel softmax dist are the output of the network
                act_n[self.agent_index] = self.gumbel_softmax_sample(logits)
            else:
                act_n[self.agent_index] = x
            q_value = self.q_network._predict_internal(obs_n + act_n)
            policy_regularization = tf.math.reduce_mean(tf.math.square(x))
            loss = -tf.math.reduce_mean(q_value) + 1e-3 * policy_regularization  # gradient plus regularization

        gradients = tape.gradient(loss, self.model.trainable_variables)  # todo not sure if this really works
        # gradients = tf.clip_by_global_norm(gradients, self.clip_norm)[0]
        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return loss


class MADDPGCriticNetwork(object):
    def __init__(self, num_hidden_layers, units_per_layer, lr, obs_n_shape, act_shape_n, act_type, agent_index):
        """
        Implementation of a critic to represent the Q-Values. Basically just a fully-connected
        regression ANN.
        """
        self.num_layers = num_hidden_layers
        self.lr = lr
        self.obs_shape_n = obs_n_shape
        self.act_shape_n = act_shape_n
        self.act_type = act_type

        self.clip_norm = 0.5
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        # set up layers
        # each agent's action and obs are treated as separate inputs
        self.obs_input_n = []
        for idx, shape in enumerate(self.obs_shape_n):
            self.obs_input_n.append(tf.keras.layers.Input(shape=shape, name='obs_in' + str(idx)))

        self.act_input_n = []
        for idx, shape in enumerate(self.act_shape_n):
            self.act_input_n.append(tf.keras.layers.Input(shape=shape, name='act_in' + str(idx)))

        self.input_concat_layer = tf.keras.layers.Concatenate()

        self.hidden_layers = []
        for idx in range(num_hidden_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}crit_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(1, activation='linear',
                                                  name='ag{}crit_out{}'.format(agent_index, idx))

        # connect layers
        x = self.input_concat_layer(self.obs_input_n + self.act_input_n)
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=self.obs_input_n + self.act_input_n,  # list concatenation
                                    outputs=[x])
        self.model.compile(self.optimizer, loss='mse')

    def predict(self, obs_n, act_n):
        """
        Predict the value of the input.
        """
        return self._predict_internal(obs_n + act_n)

    @tf.function(experimental_relax_shapes=True)
    def _predict_internal(self, concatenated_input):
        """
        Internal function, because concatenation can not be done in tf.function
        """
        x = self.input_concat_layer(concatenated_input)
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)
        return x

    def train_step(self, obs_n, act_n, target_q, weights):
        """
        Train the critic network with the observations, actions, rewards and next observations, and next actions.
        """
        return self._train_step_internal(obs_n + act_n, target_q, weights)

    @tf.function(experimental_relax_shapes=True)
    def _train_step_internal(self, concatenated_input, target_q, weights):
        """
        Internal function, because concatenation can not be done inside tf.function
        """
        with tf.GradientTape() as tape:
            x = self.input_concat_layer(concatenated_input)
            for idx in range(self.num_layers):
                x = self.hidden_layers[idx](x)
            q_pred = self.output_layer(x)
            td_loss = tf.math.square(target_q - q_pred)
            loss = tf.reduce_mean(td_loss * weights)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))

        return td_loss
