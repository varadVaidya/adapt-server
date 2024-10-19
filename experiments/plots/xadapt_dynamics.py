import os
import random
import subprocess
from dataclasses import dataclass, asdict
from typing import Union

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.sans-serif": "courier",
    }
)
SAVE_PATH = "/home/varad/robotics/adapt-figs/"
plt.style.use("seaborn-v0_8-paper")

from adapt_drones.utils.dynamics import CustomDynamics

from experiments.xadapt.scale_functions import *


seed = 15092024
rng = np.random.default_rng(seed=seed)
c = np.linspace(0, 1, 100)
xadapt_dynamics = Dynamics(seed=rng, c=c, do_random=False)

arm_lengths = xadapt_dynamics.length_scale()
mass = xadapt_dynamics.mass_scale()
ixx = xadapt_dynamics.ixx_yy_scale()
izz = xadapt_dynamics.izz_scale()
km_kf = xadapt_dynamics.torque_to_thrust()


mass_range = np.zeros((100, 2))
ixx_range = np.zeros((100, 2))
izz_range = np.zeros((100, 2))
km_kf_range = np.zeros((100, 2))


for i in range(len(c)):

    mass_min, mass_max = np.inf, -np.inf
    ixx_min, ixx_max = np.inf, -np.inf
    izz_min, izz_max = np.inf, -np.inf
    km_kf_min, km_kf_max = np.inf, -np.inf

    xadapt_dynamics = MinMaxDynamics(seed=rng, c=c[i], do_random=True)
    xadapt_dynamics.length_scale()

    mass_range[i] = xadapt_dynamics.mass_scale()
    ixx_range[i] = xadapt_dynamics.ixx_yy_scale()
    izz_range[i] = xadapt_dynamics.izz_scale()
    km_kf_range[i] = xadapt_dynamics.torque_to_thrust()

mass_range = np.array(mass_range).T
ixx_range = np.array(ixx_range).T
izz_range = np.array(izz_range).T
km_kf_range = np.array(km_kf_range).T

print(mass_range.shape)


fig, axs = plt.subplots(
    2,
    2,
    figsize=(6.5, 6.5),
    sharex=True,
    # layout="compressed",
)
ax1, ax2, ax3, ax4 = axs.flatten()
colours = {
    "train": "#7ad151",
    "eval": "#22a884",
    "border_eval": "#414487",
}


train_fill_plot_list = [
    (mass, mass_range, ax1, "Mass", "$M$", "$(kg)$", (0, 0)),
    (
        ixx,
        ixx_range,
        ax2,
        "Inertia XX (Inertia YY)",
        "$I_{xx} and I_{yy}$",
        "$(kg \ m^2)$",
        (-2, -2),
    ),
    (izz, izz_range, ax3, "Inertia ZZ", "$I_{zz}$", "$(kg \ m^2)$", (-2, -2)),
    (
        km_kf,
        km_kf_range,
        ax4,
        "Propellor Constant",
        "$K_m/K_f$",
        "$(m)$",
        (0, 0),
    ),
]

for i, (data, data_range, ax, title, label, unit, scilimit) in enumerate(
    train_fill_plot_list
):
    ax.fill_between(
        arm_lengths,
        data_range[0],
        data_range[1],
        alpha=0.9,
        color=colours["eval"],
        label="Eval Range",
        # hatch="X",
    )

    ax.fill_between(
        arm_lengths,
        data_range[0],
        data_range[1],
        alpha=1,
        color=None,
        fc="none",
        # hatch="X",
        edgecolor=colours["border_eval"],
        linewidth=1,
    )
    # ax.plot(arm_lengths, data, color="royalblue")
    ax.grid(linestyle="--")
    # ax.set_ylabel(f"{label} {unit}", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.ticklabel_format(axis="y", style="sci", scilimits=scilimit)
    # ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

ax3.set_xlabel("Arm Length (m)")
ax4.set_xlabel("Arm Length (m)")
legend = ax4.get_legend_handles_labels()
# ax6.legend(
#     legend[0],
#     legend[1],
#     loc="center",
#     fontsize=9,
#     bbox_to_anchor=(0.5, 0.5),
#     fancybox=True,
#     shadow=True,
#     ncols=2,
# )

plt.legend(
    legend[0],
    legend[1],
    loc="lower left",
    fontsize=10,
    fancybox=True,
    shadow=True,
    # orientation="horizontal",
    bbox_to_anchor=(-0.4, -0.3),
    ncols=2,
)

# plt.show()

plt.savefig(SAVE_PATH + "xadapt_dynamics.svg", bbox_inches="tight")
plt.savefig(SAVE_PATH + "xadapt_dynamics.png", bbox_inches="tight")
