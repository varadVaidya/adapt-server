import os
import random
import subprocess
from dataclasses import dataclass
import concurrent.futures

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym
import tyro

from adapt_drones.cfgs.config import *
from adapt_drones.cfgs.environment_cfg import *
from adapt_drones.utils.eval import paper_phase_1_eval, paper_RMA_DATT_eval
from adapt_drones.utils.git_utils import check_git_clean

# check_git_clean()  # TODO: see if needed in eval


@dataclass
class Args:
    env_id: str = "traj_v2"
    run_name: str = "absurd-puddle-1"
    seed: int = 20240915
    agent: str = "RMA_DATT"
    scale: bool = True


args = tyro.cli(Args)
cfg = Config(
    env_id=args.env_id,
    seed=args.seed,
    eval=True,
    run_name=args.run_name,
    agent=args.agent,
    scale=args.scale,
)

current_branch_name = (
    subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    .decode("utf-8")
    .strip()
)
print("Current branch name:", current_branch_name)
branch_name = "runs/" + cfg.experiment.grp_name + "/" + args.run_name


### * EVAL CODE

sc = np.linspace(0.05, 0.16, 12)
sc_list = [[i, i] for i in sc]
num_sc_list = len(sc_list)
print("Scale lengths:", num_sc_list)

# create a list of seeds by incrementing cfg.seed by 1
num_seeds = 10
seeds = [cfg.seed + i for i in range(num_seeds)]
print("Seeds:", seeds)

all_results = np.zeros((num_sc_list, num_seeds, 10))
# 10 : scale, seed, mean_phase_1, rms_phase_1, mean_RMA_DATT, rms_RMA_DATT,
#  mass, ixx, iyy, izz
phase_1_results = np.zeros((num_sc_list, num_seeds, 8))
rma_datt_results = np.zeros((num_sc_list, num_seeds, 8))

for _scale in sc_list:
    # for _seed in seeds:

    #     print("-----------==========++++++++++==========-----------")
    #     cfg.environment.scale_lengths = _scale
    #     cfg.scale.scale_lengths = _scale

    #     mean_e, rms_e = paper_phase_1_eval(cfg=cfg, best_model=True)
    #     print(f"Mean: {mean_e}, RMS: {rms_e}")
    #     mean_e, rms_e = paper_RMA_DATT_eval(cfg=cfg, best_model=True)
    #     print(f"Mean: {mean_e}, RMS: {rms_e}")
    #     print("-----------==========++++++++++==========-----------")
    num_cores = 4  # one for each seed
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(paper_phase_1_eval, [cfg] * num_seeds, seeds))
        # phase_1_results = np.array(results)
