import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os
import datetime
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

import maddpg.common.tf_util as U
from maddpg.trainer.trainer_KL_sw import KLTrainer
from maddpg.trainer.maddpg_sw import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=50000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=3, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="ddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="ddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy/0924/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--load-good", type=str, default="", help="directory in which training state and model of good agents are loaded")
    parser.add_argument("--load-adv", type=str, default="", help="directory in which training state and model of adv agents are loaded")
    
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="/benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="/learning_curves/0924/", help="directory where plot data is saved")
    
    #other settings
    parser.add_argument("--train-mode", action="store_true", default=False)
    parser.add_argument("--train-adv", action="store_true", default=False)
    parser.add_argument("--train-good", action="store_true", default=False)
    
    #gpu
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--mode", type=str, default="0")


    return parser.parse_args()

        
def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model

    for i in range(num_adversaries):
        if arglist.adv_policy == "kl":
            trainers.append(KLTrainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist, trainable=arglist.train_adv))
        else:
            trainers.append(MADDPGAgentTrainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func = (arglist.adv_policy == "ddpg"), trainable=arglist.train_adv))

    for i in range(num_adversaries, env.n):
        if arglist.good_policy == "kl":
            trainers.append(KLTrainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist, trainable=arglist.train_good))
        else:
            trainers.append(MADDPGAgentTrainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func= (arglist.good_policy == "ddpg"), trainable=arglist.train_good))
    
    return trainers


def train(arglist):
    os.environ["CUDA_VISIBLE_DEVICES"] = arglist.gpu
    with U.single_threaded_session() as sess:
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()
        
        # Create tensorboard summary writer
        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "./logs/0924/" + arglist.exp_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # log_dir = "./logs/0421/" + arglist.exp_name
        
        sw = tf.summary.create_file_writer(log_dir)
        with sw.as_default():
            tf.summary.text("Arguments List: ", str(arglist), step=0)
            sw.flush()


        # sw = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
        # summary_op = tf.summary.text("Arguments List: ", tf.convert_to_tensor(str(arglist)))
        # summary = sess.run(summary_op)
        # sw.add_summary(summary)

        arglist.save_dir = arglist.save_dir + arglist.exp_name + '/'
        # Load previous results, if necessary
        if arglist.load_dir != "":
            print('Loading previous state...')
            U.load_state(arglist.load_dir) 
        #     arglist.load_dir = arglist.save_dir
        # if arglist.display or arglist.restore or arglist.benchmark or not arglist.train_mode:
            # print('Loading previous state...')
            # U.load_state(arglist.load_dir)       
            
        if arglist.load_adv != "":
            print('Loading previous state of advesaries...')
            for i in range(num_adversaries):
                U.load_part_state(arglist.load_adv, "agent_%d" % i)
        if arglist.load_good != "":
            print('Loading previous state of good agents...')
            for i in range(num_adversaries, env.n):
                U.load_part_state(arglist.load_good, "agent_%d" % i)


        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.compat.v1.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        
        print('Starting iterations...')
        while True:
            # get action
            action_n = []
            
            for i in range(env.n):
                if trainers[i].maddpg_mode:
                    action_n.append(trainers[i].action(obs_n[i])) if trainers[i].trainable else action_n.append(trainers[i].test_action(obs_n[i]))
                else:
                    action_n.append(trainers[i].full_train_action(obs_n) if arglist.train_mode else trainers[i].local_test_action(obs_n[i]))
            
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update trainable trainers, if not in display or benchmark mode
            if arglist.train_mode:
                for agent in trainers:
                    if agent.trainable:
                        loss = None
                        agent.preupdate()
                        loss = agent.update(trainers, train_step, sw)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                if arglist.train_mode:
                    U.save_state(arglist.save_dir, len(episode_rewards), saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)