import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


"""Actor (Policy) Model."""
#KL p_train
def p_train(make_obs_ph_n, act_space_n, p_index, full_p_func, q_func, local_p_func, optimizer, grad_norm_clipping=None, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        # act_pdtype_n = []
        # for i, act_space in enumerate(act_space_n):
        #     print("agent {}:".format(i))
        #     act_pdtype_n.append(make_pdtype(act_space))
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # exit(0)
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        
        act_input_n = act_ph_n + []
        # act_input_n = act_ph_n
        full_p_input = tf.concat(obs_ph_n, 1)
        local_p_input = obs_ph_n[p_index]
        
        # print(int(act_pdtype_n[p_index].param_shape()[0]))
        local_p = local_p_func(local_p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="local_p_func", num_units=num_units)
        full_p = full_p_func(full_p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="full_p_func", num_units=num_units)
        
        full_p_func_vars = U.scope_vars(U.absolute_scope_name("full_p_func"))
        local_p_func_vars = U.scope_vars(U.absolute_scope_name("local_p_func"))
        
        '''full loss'''
        # wrap parameters in distribution
        full_act_pd = act_pdtype_n[p_index].pdfromflat(full_p)
        full_train_act_sample = full_act_pd.sample()
        full_test_act_sample = full_act_pd.nobias_sample()
        full_p_reg = tf.reduce_mean(tf.square(full_act_pd.flatparam()))
        
        act_input_n[p_index] = full_train_act_sample
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        full_loss = pg_loss + full_p_reg * 1e-3
        
        '''local loss calculation'''
        local_act_pd = act_pdtype_n[p_index].pdfromflat(local_p)
        local_train_act_sample = local_act_pd.sample()
        local_test_act_sample = local_act_pd.nobias_sample()
        local_p_reg = tf.reduce_mean(tf.square(local_act_pd.flatparam()))
        
        act_input_n[p_index] = local_train_act_sample
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        q_ = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        local_pg_loss = -tf.reduce_mean(q_)
        
        # target network
        target_p = full_p_func(full_p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(full_p_func_vars, target_p_func_vars)
        target_act_pd = act_pdtype_n[p_index].pdfromflat(target_p)
        
        '''KL divergence between the targe p and local p'''
        KL_loss = tf.reduce_mean(target_act_pd.kl(local_act_pd))
        
        #original
        # local_loss = local_pg_loss + local_p_reg * 1e-3 + KL_loss * 0.1
        local_loss = local_pg_loss + local_p_reg * 1e-3 + KL_loss * 0.01
        
        # only pg
        # local_loss = local_pg_loss + local_p_reg * 1e-3
        #only kl
        # local_loss = KL_loss * 0.1
        #new
        # local_loss = local_pg_loss + local_p_reg * 1e-3 + KL_loss * 1e-3
        
        # local_loss = KL_loss 

        full_optimize_expr = U.minimize_and_clip(optimizer, full_loss, full_p_func_vars, grad_norm_clipping)
        local_optimize_expr = U.minimize_and_clip(optimizer, local_loss, local_p_func_vars, grad_norm_clipping)

        # Create callable functions
        full_train = U.function(inputs=obs_ph_n + act_ph_n, outputs=full_loss, updates=[full_optimize_expr])
        local_train = U.function(inputs=obs_ph_n + act_ph_n, outputs=local_loss, updates=[local_optimize_expr])
        
        full_train_act = U.function(inputs= obs_ph_n, outputs=full_train_act_sample)
        local_train_act = U.function(inputs=[obs_ph_n[p_index]], outputs=local_train_act_sample)
        full_test_act = U.function(inputs=obs_ph_n, outputs=full_test_act_sample)
        local_test_act = U.function(inputs=[obs_ph_n[p_index]], outputs=local_test_act_sample)
        
        p_values = []
        p_values.append(full_p)
        p_values.append(target_p)
        p_values.append(local_p)
        p_values.append(KL_loss)
        
        p_values_func = U.function(obs_ph_n, p_values)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=obs_ph_n, outputs=target_act_sample)
        
        # create summary
        summary = []
        summary.append(tf.summary.scalar('full_loss', full_loss))
        summary.append(tf.summary.scalar('pg_loss', pg_loss))
        summary.append(tf.summary.scalar('full_p_reg', full_p_reg))
        summary.append(tf.summary.scalar('local_loss', local_loss))
        summary.append(tf.summary.scalar('KL_loss', KL_loss))
        summary.append(tf.summary.scalar('local_pg_loss', local_pg_loss))
        summary.append(tf.summary.scalar('local_p_reg', local_p_reg))
        return_summary = U.function(inputs= obs_ph_n + act_ph_n, outputs=summary)
        
        return full_train_act, full_test_act, local_train_act, local_test_act, full_train, local_train, update_target_p, {'p_values_func': p_values_func, 'target_act': target_act}, return_summary

    
"""Critic (Value) Model."""
def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)
        
        # create summary
        summary = []
        summary.append(tf.summary.scalar('q_loss', loss))
        return_summary = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=summary)
        
        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}, return_summary

class KLTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, trainable=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.maddpg_mode = False
        self.trainable = trainable
        # create place_holder for obsevation
        
        print("agent{} Trainable: {}".format(agent_index, self.trainable))

        obs_ph_n = []
        for i in range(len(obs_shape_n)):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())
        
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug, self.q_summary = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units
        )
        self.full_train_act, self.full_test_act, self.local_train_act, self.local_test_act, self.full_p_train, self.local_p_train, self.p_update, self.p_debug, self.p_summary = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            full_p_func=model,
            q_func=model,
            local_p_func = model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    #input: full observation
    def full_train_action(self, obs):
        new_obs = []
        for obs_ in obs:
            new_obs.append(obs_.reshape(1, -1))
        return self.full_train_act(*(new_obs))[0]
        # return self.full_act(np.expand_dims(obs, axis = 0))[0]

    def full_test_action(self, obs):
        new_obs = []
        for obs_ in obs:
            new_obs.append(obs_.reshape(1, -1))
        return self.full_test_ct(*(new_obs))[0]

    #input: local observation
    def local_train_action(self, obs):
        return self.local_train_act(obs[None])[0]

    def local_test_action(self, obs):
        return self.local_test_act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t, sw):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for j in range(num_sample):
            # target_act_next_n = []
            # for i in range(self.n):
            #     if agents[i].maddpg_mode is True:
            #         target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i])
            #     else:
            #         target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n)
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) if agents[i].maddpg_mode is True else agents[i].p_debug['target_act'](*(obs_next_n)) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        full_p_loss = self.full_p_train(*(obs_n + act_n))
        local_p_loss = self.local_p_train(*(obs_n + act_n))
        
        self.p_update()
        self.q_update()
        
        if t % self.args.save_rate == 0:
            p_loss_summary = self.p_summary(*(obs_n + act_n))
            for summary in p_loss_summary:
                sw.add_summary(summary, t) 
            q_loss_summary = self.q_summary(*(obs_n + act_n + [target_q]))
            for summary in q_loss_summary:
                sw.add_summary(summary, t) 
        # p_values = agents[self.agent_index].p_debug['p_values_func'](*obs_n)
        # print(p_values[0])
        # print(p_values[1])
        # print(p_values[2])
        # print(p_values[3])
        # print(agents[self.agent_index].p_debug['full_p_value'](*obs_n))
        # print(agents[self.agent_index].p_debug['local_p_value'](obs_n[self.agent_index]))
        return [q_loss, full_p_loss, local_p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]