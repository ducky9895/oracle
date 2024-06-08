#!/bin/bash

# Ensure the script is run from the directory containing this script
cd "$(dirname "$0")"

# Ensure the config.json file exists
CONFIG_FILE="config.json"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "config.json file not found!"
  exit 1
fi

# Install jq if not already installed
if ! command -v jq &> /dev/null
then
    echo "jq could not be found, installing..."
    sudo apt-get update
    sudo apt-get install -y jq
fi

# Create directory for evaluation results if it does not exist
mkdir -p ../evaluation_results

# Read scenarios and experiments from the config.json file
SCENARIOS=$(jq -r '.scenarios | to_entries[] | .key' "$CONFIG_FILE")
EXPERIMENTS=$(jq -c '.experiments[]' "$CONFIG_FILE")

# Define evaluation function
run_evaluation () {
  local scenario=$1
  local good_policy=$2
  local adv_policy=$3
  local exp_name="${good_policy}_${adv_policy}_${scenario}"

  # Get number of agents and adversaries from the config
  local num_agents=$(jq -r --arg scenario "$scenario" '.scenarios[$scenario].agents' "$CONFIG_FILE")
  local num_adversaries=$(jq -r --arg scenario "$scenario" '.scenarios[$scenario].adversaries' "$CONFIG_FILE")

  # Run the evaluation script with the specified policies and configurations
  python test_KL_mode.py --exp-name $exp_name --scenario $scenario --adv-policy $adv_policy --good-policy $good_policy --num-adversaries $num_adversaries
  mv ./${exp_name}_evaluation.pkl ../evaluation_results/${good_policy}_${adv_policy}_${scenario}_evaluation.pkl
}

export -f run_evaluation

# Prepare the commands to run in parallel
commands=()
for SCENARIO in $SCENARIOS; do
  for EXPERIMENT in $EXPERIMENTS; do
    GOOD_POLICY=$(echo $EXPERIMENT | jq -r '.good_policy')
    ADV_POLICY=$(echo $EXPERIMENT | jq -r '.adv_policy')
    commands+=("run_evaluation $SCENARIO $GOOD_POLICY $ADV_POLICY")
  done
done

# Run the commands in parallel
printf "%s\n" "${commands[@]}" | parallel

echo "All done with evaluation!"
