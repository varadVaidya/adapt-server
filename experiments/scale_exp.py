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
    env_id: str = "traj_v2_ctbr"
    run_name: str = "zany-elevator-1"
    seed: int = 4551
    agent: str = "RMA_DATT"
    scale: bool = True


env_runs = [["traj_v2", "fiery-pine-13"], ["traj_v2", "leafy-wood-11"]]

for env_run in env_runs:
    args = Args(env_id=env_run[0], run_name=env_run[1])
    print("Running for env:", args.env_id, "run_name:", args.run_name)

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

    sc = np.linspace(0.05, 0.16, 11)
    print("Scale lengths:", sc)
    sc_list = [[i, i] for i in sc]
    num_sc_list = len(sc_list)
    print("Scale lengths:", num_sc_list)

    # create a list of seeds by incrementing cfg.seed by 1
    num_seeds = 50
    seeds = [cfg.seed + i for i in range(num_seeds)]
    print("Seeds:", seeds)

    all_results = np.zeros((num_sc_list, num_seeds, 10))
    # 8 : scale, seed, mean_e, rms_e,  mass, ixx, iyy, izz
    phase_1_results = np.zeros((num_seeds, num_sc_list, 8))
    rma_datt_results = np.zeros((num_seeds, num_sc_list, 8))

    current_results = np.zeros(8)

    for i, _seed in enumerate(seeds):
        for j, _scale in enumerate(sc_list):
            cfg.seed = _seed
            cfg.environment.scale_lengths = _scale
            cfg.scale.scale_lengths = _scale
            # print("-----------==========++++++++++==========-----------")
            results = paper_phase_1_eval(cfg=cfg, best_model=True)
            current_results[0] = _scale[0]
            current_results[1] = _seed
            current_results[2:] = results

            phase_1_results[i, j, :] = current_results
            # print("-----------==========++++++++++==========-----------")

            results = paper_RMA_DATT_eval(cfg=cfg, best_model=True)
            current_results[0] = _scale[0]
            current_results[1] = _seed
            current_results[2:] = results

            rma_datt_results[i, j, :] = current_results

    run_folder = "runs/" + cfg.grp_name + "/" + cfg.run_name + "/"
    results_folder = run_folder + "results-icra/"

    os.makedirs(results_folder, exist_ok=True)

    print("Saving results to:", results_folder)
    np.save(results_folder + "phase_1_results.npy", phase_1_results)
    np.save(results_folder + "rma_datt_results.npy", rma_datt_results)
