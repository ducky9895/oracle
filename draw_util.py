import os
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Updated mapping of environments to agent names and number of adversaries
environment_agents = {
    'simple_adversary': (['adversary_0', 'adversary_1', 'adversary_2', 'agent_0', 'agent_1'], 3),
    'simple_crypto': (['eve', 'alice', 'bob'], 1),
    'simple_push': (['adversary_0', 'agent_0'], 1),
    'simple_speaker_listener': (['speaker', 'listener'], 0),
    'simple_spread': (['agent_0', 'agent_1', 'agent_2'], 0),
    'simple_tag': (['adversary_0', 'adversary_1', 'adversary_2', 'agent_0', 'agent_1'], 3),
    'simple_world_comm': (['agent_0', 'agent_1', 'agent_2', 'agent_3'], 0),
    'simple_reference': (['agent_0', 'agent_1'], 0),
    'simple_v2': (['agent_0', 'agent_1'], 0)
}

def get_unique_filename(directory, base_name, extension):
    """
    Generate a unique file name by appending a numerical suffix if needed.
    """
    counter = 1
    file_name = f"{base_name}{extension}"
    while os.path.exists(os.path.join(directory, file_name)):
        file_name = f"{base_name}-{counter}{extension}"
        counter += 1
    return file_name

def plot_metrics(run_dirs, scenario, figures_dir):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
    axs = axs.flatten()
    
    for i, run_dir in enumerate(run_dirs):
        rewards_file = os.path.join(run_dir, 'rewards.pkl')
        agrewards_file = os.path.join(run_dir, 'agrewards.pkl')
        args_file = os.path.join(run_dir, 'args.pkl')

        if not os.path.exists(rewards_file):
            print(f"No rewards found in {run_dir}, skipping...")
            continue

        with open(rewards_file, 'rb') as f:
            rewards = pickle.load(f)

        agrewards = None
        if os.path.exists(agrewards_file):
            with open(agrewards_file, 'rb') as f:
                agrewards = pickle.load(f)

        with open(args_file, 'rb') as f:
            args = pickle.load(f)
            exp_name = args.get('exp_name', 'Experiment')
            scenario_name = args.get('scenario_name', 'unknown_scenario')

        if not rewards:
            print(f"No data found in {run_dir}, skipping...")
            continue

        agent_names, num_adversaries = environment_agents.get(scenario_name, ([], 0))
        num_good_agents = len(agent_names) - num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_episodes = len(rewards) // num_agents
        episodes = list(range(num_episodes))

        for j in range(num_adversaries):
            agent_rewards = rewards[j::num_agents]
            agent_name = agent_names[j] if j < len(agent_names) else f'Adversary {j+1}'
            axs[i].plot(episodes, agent_rewards[:num_episodes], label=f'{agent_name}')

        if agrewards:
            for j in range(num_good_agents):
                agent_agrewards = agrewards[j::num_agents]
                agent_name = agent_names[num_adversaries + j] if (num_adversaries + j) < len(agent_names) else f'Agent {j+1}'
                axs[i].plot(episodes, agent_agrewards[:num_episodes], label=f'{agent_name}')

        axs[i].set_title(exp_name)
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel('Rewards')
        axs[i].legend()

    plt.tight_layout()
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    base_file_name = f"{scenario}_training_progress"
    unique_file_name = get_unique_filename(figures_dir, base_file_name, ".png")

    plt.savefig(os.path.join(figures_dir, unique_file_name))
    plt.close()
    print(f"Plot saved for {scenario} as {unique_file_name}")

def plot_all_results(base_dir='results/1', figures_dir='results/figures/1'):
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    for scenario in environment_agents.keys():
        scenario_dir = os.path.join(base_dir, scenario)
        if os.path.isdir(scenario_dir):
            run_dirs = [os.path.join(scenario_dir, run) 
                        for run in sorted(os.listdir(scenario_dir))]
            if run_dirs:
                plot_metrics(run_dirs, scenario, figures_dir)

if __name__ == '__main__':
    plot_all_results()
