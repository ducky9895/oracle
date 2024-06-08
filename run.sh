#!/bin/bash

# Define the scenarios
scenarios=(
  "simple_adversary"
  "simple_crypto"
  "simple_push"
  "simple_reference"
  "simple_speaker_listener"
  "simple_spread"
  "simple_tag"
  "simple_world_comm"
)

# Define other parameters
num_episodes=60000
max_episode_len=25
batch_size=1024
lr=1e-2
gamma=0.95
num_layers=2
num_units=64
update_rate=100
critic_zero_if_done=False
buff_size=1e6
tau=0.01
hard_max=False
priori_replay=False
alpha=0.6
beta=0.5
use_target_action=True

# Create results directory if it does not exist
mkdir -p results

# Function to run an experiment
run_experiment() {
  local scenario=$1
  local good_policy=$2
  local adv_policy=$3
  local exp_name="exp_${scenario}_${good_policy}_vs_${adv_policy}"

  # Run the experiment using Sacred
  python train.py with exp_name=$exp_name scenario_name=$scenario num_episodes=$num_episodes \
    max_episode_len=$max_episode_len batch_size=$batch_size lr=$lr gamma=$gamma num_layers=$num_layers \
    num_units=$num_units update_rate=$update_rate critic_zero_if_done=$critic_zero_if_done \
    buff_size=$buff_size tau=$tau hard_max=$hard_max priori_replay=$priori_replay alpha=$alpha \
    beta=$beta use_target_action=$use_target_action good_policy=$good_policy adv_policy=$adv_policy \
    save_rate=100 restore_fp=None
}

# Loop over each scenario and run experiments
for scenario in "${scenarios[@]}"; do
  echo "Running experiments for scenario: $scenario"

  # Run all combinations of good_policy and adv_policy
  run_experiment $scenario "maddpg" "maddpg"
  run_experiment $scenario "maddpg" "maddpgkl"
  run_experiment $scenario "maddpgkl" "maddpg"
  run_experiment $scenario "maddpgkl" "maddpgkl"
done

echo "All experiments completed."
