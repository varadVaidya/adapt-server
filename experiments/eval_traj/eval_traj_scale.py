import os
import random
import subprocess
from dataclasses import dataclass
import concurrent.futures
from itertools import repeat
import time

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
    env_id: str
    run_name: str
    seed: int = 4551
    agent: str = "RMA_DATT"
    scale: bool = True
    wind_bool: bool = True


env_runs = [
    ["traj_v3", "laced-fire-32", False],
    ["traj_v3", "laced-fire-32", True],
]


def evaluate_per_seed(seed, num_sc_list, sc_list, cfg, idx):
    # print("Evaluating idx:", idx, "seed:", seed, "at time:", time.asctime())
    phase_1_seed_results = np.zeros((num_sc_list, 9))
    rma_datt_seed_results = np.zeros((num_sc_list, 9))
    for i, _scale in enumerate(sc_list):
        print(
            "Evaluating idx:",
            idx,
            "seed:",
            seed,
            "scale:",
            _scale,
            "at time:",
            time.asctime(),
        )
        phase_1_current_results = np.zeros(9)
        rma_datt_current_results = np.zeros(9)
        cfg.seed = seed
        cfg.environment.scale_lengths = _scale
        cfg.scale.scale_lengths = _scale
        # print("-----------==========++++++++++==========-----------")
        results = paper_phase_1_eval(
            cfg=cfg, best_model=True, idx=idx, return_traj_len=True
        )
        phase_1_current_results[0] = _scale[0]
        phase_1_current_results[1] = seed
        phase_1_current_results[2:] = results

        phase_1_seed_results[i, :] = phase_1_current_results

        # print("-----------==========++++++++++==========-----------")

        results = paper_RMA_DATT_eval(
            cfg=cfg, best_model=True, idx=idx, return_traj_len=True
        )
        rma_datt_current_results[0] = _scale[0]
        rma_datt_current_results[1] = seed
        rma_datt_current_results[2:] = results

        rma_datt_seed_results[i, :] = rma_datt_current_results
        print(
            "Done evaluating idx:",
            idx,
            "seed:",
            seed,
            "scale:",
            _scale,
            "at time:",
            time.asctime(),
        )

    return seed, phase_1_seed_results, rma_datt_seed_results


def trajectory_eval_idx(idx, num_seeds, seeds, num_sc_list, sc_list, cfg):
    print("Evalating trajectory idx:", idx)
    # 8 : scale, seed, mean_e, rms_e,  mass, ixx, iyy, izz
    phase_1_results = np.zeros((num_seeds, num_sc_list, 9))
    rma_datt_results = np.zeros((num_seeds, num_sc_list, 9))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            evaluate_per_seed,
            seeds,
            repeat(num_sc_list),
            repeat(sc_list),
            repeat(cfg),
            repeat(idx),
        )

        for i, (seed, phase_1, rma_datt) in enumerate(results):
            phase_1_results[i] = phase_1
            rma_datt_results[i] = rma_datt

    return idx, phase_1_results, rma_datt_results


for env_run in env_runs:
    args = Args(env_id=env_run[0], run_name=env_run[1], wind_bool=env_run[2])
    print(
        "Running for env:",
        args.env_id,
        "run_name:",
        args.run_name,
        "wind_bool:",
        args.wind_bool,
    )

    cfg = Config(
        env_id=args.env_id,
        seed=args.seed,
        eval=True,
        run_name=args.run_name,
        agent=args.agent,
        scale=args.scale,
        wind_bool=args.wind_bool,
    )

    current_branch_name = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    print("Current branch name:", current_branch_name)
    branch_name = "runs/" + cfg.experiment.grp_name + "/" + args.run_name

    ### * EVAL CODE

    sc = np.linspace(0.05, 0.20, 16)
    print("Scale lengths:", sc)
    sc_list = [[i, i] for i in sc]
    num_sc_list = len(sc_list)
    print("Scale lengths:", num_sc_list)

    num_eval_trajs = 13  # 13 eval trajs #& HARDCODED for now
    print("Number of eval trajs:", num_eval_trajs)

    # create a list of seeds by incrementing cfg.seed by 1
    num_seeds = 16
    seeds = [cfg.seed + i for i in range(num_seeds)]
    print("Seeds:", seeds)

    all_phase_1_results = np.zeros((num_eval_trajs, num_seeds, num_sc_list, 9))
    all_rma_datt_results = np.zeros((num_eval_trajs, num_seeds, num_sc_list, 9))

    # set all values to nan for better checking if all values are filled
    all_phase_1_results[:] = np.nan
    all_rma_datt_results[:] = np.nan

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        results = executor.map(
            trajectory_eval_idx,
            range(num_eval_trajs),
            repeat(num_seeds),
            repeat(seeds),
            repeat(num_sc_list),
            repeat(sc_list),
            repeat(cfg),
        )

        for idx, phase_1_results, rma_datt_results in results:
            print("idx:", idx)
            all_phase_1_results[idx] = phase_1_results
            all_rma_datt_results[idx] = rma_datt_results

    run_folder = (
        "runs/"
        + cfg.experiment.wandb_project_name
        + "/"
        + cfg.grp_name
        + "/"
        + cfg.run_name
        + "/"
    )
    results_folder = run_folder + "results-icra/"

    os.makedirs(results_folder, exist_ok=True)

    print("Saving results to:", results_folder)
    prefix = "wind_" if args.wind_bool else "no_wind_"
    # check if all values are filled
    print(
        "Nan present in phase_1_results:",
        np.isnan(all_phase_1_results).any(),
        "Nan present in rma_datt_results:",
        np.isnan(all_rma_datt_results).any(),
    )
    np.save(results_folder + prefix + "phase_1_traj.npy", all_phase_1_results)
    np.save(results_folder + prefix + "rma_datt_traj.npy", all_rma_datt_results)
