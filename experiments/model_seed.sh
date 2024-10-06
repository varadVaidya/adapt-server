#!/bin/bash

# Define the range of seeds
seq 4551 4560 | xargs -P 2 -I {} python adapt_drones/train.py --env_id traj_v3 --seed {}
