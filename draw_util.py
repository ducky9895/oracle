#%%
import matplotlib.pyplot as plt
import pickle
import json
#%%
def load_rewards(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_rewards(agent_rewards, title, labels, save_path):
    plt.figure(figsize=(10, 7))
    for i, rewards in enumerate(agent_rewards):
        plt.plot(rewards, label=labels[i])
    plt.xlabel('Episodes (1000)')
    plt.ylabel('Agent Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    scenarios = config.keys()
    for scenario in scenarios:
        agents = config[scenario]["agents"]

        experiments = [
            ('kl_vs_ddpg_adversary', f'kl_ddpg_{scenario}_rewards.pkl', agents),
            ('maddpg_vs_ddpg_adversary', f'maddpg_ddpg_{scenario}_rewards.pkl', agents),
            ('ddpg_vs_kl_agent', f'ddpg_kl_{scenario}_rewards.pkl', agents),
            ('ddpg_vs_maddpg_agent', f'ddpg_maddpg_{scenario}_rewards.pkl', agents)
        ]

        for title, reward_file, labels in experiments:
            rewards = load_rewards(reward_file)
            plot_rewards(rewards, f'{scenario}: {title}', labels, f'{title}_{scenario}.png')

if __name__ == "__main__":
    main()
