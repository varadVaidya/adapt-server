import os
import random
import subprocess
from dataclasses import dataclass, asdict
from typing import Union

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym
import tyro

from adapt_drones.cfgs.config import *
from adapt_drones.utils.eval import phase1_eval, RMA_DATT_eval
from adapt_drones.utils.eval import paper_phase_1_eval, paper_RMA_DATT_eval
from adapt_drones.utils.git_utils import check_git_clean
from adapt_drones.utils.dynamics import CustomDynamics

from experiments.xadapt.scale_functions import *

import concurrent.futures
import tqdm as tqdm

# check_git_clean()  # TODO: see if needed in eval


@dataclass
class Args:
    env_id: str = "traj_v3"
    run_name: str = "laced-fire-32"
    seed: int = 15092024
    agent: str = "RMA_DATT"
    scale: bool = True
    idx: Union[int, None] = None
    wind_bool: bool = False


args = tyro.cli(Args)
cfg = Config(
    env_id=args.env_id,
    seed=args.seed,
    eval=True,
    run_name=args.run_name,
    agent=args.agent,
    scale=args.scale,
    wind_bool=args.wind_bool,
)


def simulate_traj_rma(idx, seed, c):
    mean_error = 0
    rms_error = 0

    rng = np.random.default_rng(seed=seed)

    xadapt_dynamics = Dynamics(seed=rng, c=c)

    dynamics = CustomDynamics(
        arm_length=xadapt_dynamics.length_scale(),
        mass=xadapt_dynamics.mass_scale(),
        ixx=xadapt_dynamics.ixx_yy_scale(),
        iyy=xadapt_dynamics.ixx_yy_scale(),
        izz=xadapt_dynamics.izz_scale(),
        km_kf=xadapt_dynamics.torque_to_thrust(),
    )

    results = paper_RMA_DATT_eval(
        cfg=cfg, best_model=True, idx=idx, options=asdict(dynamics)
    )

    mean_error, rms_error = results[:2]

    return idx, seed, c, mean_error, rms_error


if __name__ == "__main__":
    c = np.linspace(0, 1, 15)
    seeds = np.arange(4551, 4551 + 16)
    idx = np.arange(0, 13)

    print(c)
    print(seeds)
    print(idx)

    num_lengths = len(c)
    num_seeds = len(seeds)
    num_idx = len(idx)

    traj_rma_eval = np.zeros((num_idx, num_seeds, num_lengths, 5))
    traj_rma_eval[:, :, :, :] = np.nan

    map_iterable = [
        (int(i), int(seed), float(_c)) for i in idx for seed in seeds for _c in c
    ]
    print(len(map_iterable))

    with concurrent.futures.ProcessPoolExecutor() as executor:

        results = list(
            tqdm.tqdm(
                executor.map(
                    simulate_traj_rma,
                    *zip(*map_iterable),
                ),
                total=len(map_iterable),
            )
        )

    for result in results:
        _idx, _seed, _c, _mean_error, _rms_error = result
        idx_idx = np.where(idx == _idx)[0][0]
        seed_idx = np.where(seeds == _seed)[0][0]
        c_idx = np.where(c == _c)[0][0]

        traj_rma_eval[idx_idx, seed_idx, c_idx, :] = [
            _idx,
            _seed,
            _c,
            _mean_error,
            _rms_error,
        ]

    print(traj_rma_eval.shape)

    # check for nans
    print(np.argwhere(np.isnan(traj_rma_eval)))
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

    np.save(results_folder + "xadapt_traj_rma.npy", traj_rma_eval)

    traj_rma_eval_csv = traj_rma_eval.reshape(-1, traj_rma_eval.shape[-1])
    print(traj_rma_eval_csv.shape)

    header = ["idx", "seed", "c", "mean_error", "rms_error"]

    np.savetxt(
        results_folder + "xadapt_traj_rma.csv",
        traj_rma_eval_csv,
        delimiter=",",
        header=",".join(header),
        comments="",
    )
