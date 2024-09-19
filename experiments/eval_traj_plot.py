import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

# latex settings for matplotlib
from dataclasses import dataclass
import tyro
from adapt_drones.cfgs.config import *


@dataclass
class Args:
    env_id: str = "traj_v3"
    run_name: str = "sweet-feather-28"
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

phase_1_results = np.load(results_folder + "wind_phase_1_traj.npy")


idx_sort_phase = np.argsort(np.mean(phase_1_results[:, :, :, 2], axis=2))
sorted_phase_1 = phase_1_results[:, idx_sort_phase[0], :, :]

phase_1 = sorted_phase_1[:, 3:-3, :, :]  # remove the top 3 and bottom 3 seeds

RMA_DATT_results = np.load(results_folder + "wind_rma_datt_traj.npy")
idx_sort_rma = np.argsort(np.mean(RMA_DATT_results[:, :, :, 2], axis=2))
sorted_rma = RMA_DATT_results[:, idx_sort_rma[0], :, :]

rma_datt = sorted_rma[:, 3:-3, :, :]  # remove the top 3 and bottom 3 seeds

print(phase_1.shape, rma_datt.shape)

idx, traj_len, scale = [], [], []
phase_1_mean, phase_1_std, phase_1_rms, phase_1_rms_std = [], [], [], []
rma_datt_mean, rma_datt_std, rma_datt_rms, rma_datt_rms_std = [], [], [], []

# compile data for phase 1
# 0: idx, 1: traj_len, 2: scale, 3: mean, 4: std, 5: rms, 6: rms_std
data_compile_phase_1 = np.zeros((phase_1.shape[0] * phase_1.shape[2], 7))

data_compile_rma_datt = np.zeros((rma_datt.shape[0] * rma_datt.shape[2], 7))

# fill in data for phase 1
# tile the idx, traj_len across axis 1, based on the number of scales
data_compile_phase_1[:, 0] = np.repeat(np.arange(phase_1.shape[0]), phase_1.shape[2])
data_compile_phase_1[:, 1] = np.repeat(phase_1[:, 0, 0, -1], phase_1.shape[2])
data_compile_phase_1[:, 2] = np.tile(phase_1[0, 0, :, 0], phase_1.shape[0])
data_compile_phase_1[:, 3] = np.mean(phase_1[:, :, :, 2], axis=1).reshape(
    phase_1.shape[0] * phase_1.shape[2]
)
data_compile_phase_1[:, 4] = np.std(phase_1[:, :, :, 2], axis=1).reshape(
    phase_1.shape[0] * phase_1.shape[2]
)
data_compile_phase_1[:, 5] = np.mean(phase_1[:, :, :, 3], axis=1).reshape(
    phase_1.shape[0] * phase_1.shape[2]
)
data_compile_phase_1[:, 6] = np.std(phase_1[:, :, :, 3], axis=1).reshape(
    phase_1.shape[0] * phase_1.shape[2]
)


data_compile_rma_datt[:, 0] = np.repeat(np.arange(rma_datt.shape[0]), rma_datt.shape[2])
data_compile_rma_datt[:, 1] = np.repeat(rma_datt[:, 0, 0, -1], rma_datt.shape[2])
data_compile_rma_datt[:, 2] = np.tile(rma_datt[0, 0, :, 0], rma_datt.shape[0])
data_compile_rma_datt[:, 3] = np.mean(rma_datt[:, :, :, 2], axis=1).reshape(
    rma_datt.shape[0] * rma_datt.shape[2]
)
data_compile_rma_datt[:, 4] = np.std(rma_datt[:, :, :, 2], axis=1).reshape(
    rma_datt.shape[0] * rma_datt.shape[2]
)
data_compile_rma_datt[:, 5] = np.mean(rma_datt[:, :, :, 3], axis=1).reshape(
    rma_datt.shape[0] * rma_datt.shape[2]
)
data_compile_rma_datt[:, 6] = np.std(rma_datt[:, :, :, 3], axis=1).reshape(
    rma_datt.shape[0] * rma_datt.shape[2]
)

print("Saving data to:", results_folder)
np.save(results_folder + "traj_eval_compile_phase_1.npy", data_compile_phase_1)
np.savetxt(
    results_folder + "traj_eval_compile_phase_1.csv",
    data_compile_phase_1,
    delimiter=",",
)
np.save(results_folder + "traj_eval_compile_rma_datt.npy", data_compile_rma_datt)
np.savetxt(
    results_folder + "traj_eval_compile_rma_datt.csv",
    data_compile_rma_datt,
    delimiter=",",
)
