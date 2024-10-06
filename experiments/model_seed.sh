#!/bin/bash

# Ensure the 'runs' directory exists
mkdir -p runs/model_seed

# Define the range of seeds
seq 4551 4560 | xargs -P 25 -I {} bash -c 'python adapt_drones/eval.py --env_id traj_v3 --run_name true-durian-33 --seed {} > runs/model_seed/eval_seed_{}.txt 2>&1'
