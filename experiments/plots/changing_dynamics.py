import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.sans-serif": "courier",
    }
)
SAVE_PATH = "/home/varad/robotics/adapt-figs/"

plt.style.use("seaborn-v0_8-paper")
import os


from dataclasses import dataclass
import tyro
from adapt_drones.cfgs.config import *
import matplotlib.gridspec as gridspec


@dataclass
class Args:
    env_id: str = "traj_v3"
    run_name: str = "earthy-snowball-77"
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


## --------------- TRAIN RANGE PLOT ----------------- ##
_sc_lengths = cfg.environment.scale_lengths
print(_sc_lengths)
arm_lengths = np.linspace(_sc_lengths[0], _sc_lengths[1], 100)

mass = np.polyval(cfg.scale.avg_mass_fit, arm_lengths)
mass_std = np.polyval(cfg.scale.std_mass_fit, arm_lengths)
mass_range = np.array([mass - mass_std, mass + mass_std])
mass_range[mass_range < 0] = 0

ixx = np.polyval(cfg.scale.avg_ixx_fit, arm_lengths)
ixx_std = np.polyval(cfg.scale.std_ixx_fit, arm_lengths)
ixx_range = np.array([ixx - ixx_std, ixx + ixx_std])
ixx_range[ixx_range < 0] = 0

iyy = np.polyval(cfg.scale.avg_iyy_fit, arm_lengths)
iyy_std = np.polyval(cfg.scale.std_iyy_fit, arm_lengths)
iyy_range = np.array([iyy - iyy_std, iyy + iyy_std])
iyy_range[iyy_range < 0] = 0

izz = np.polyval(cfg.scale.avg_izz_fit, arm_lengths)
izz_std = np.polyval(cfg.scale.std_izz_fit, arm_lengths)
izz_range = np.array([izz - izz_std, izz + izz_std])
izz_range[izz_range < 0] = 0

km_kf = np.polyval(cfg.scale.avg_km_kf_fit, arm_lengths)
km_kf_std = np.polyval(cfg.scale.std_km_kf_fit, arm_lengths)
while np.any(km_kf - km_kf_std < 5e-4):
    km_kf_std[np.where(km_kf - km_kf_std < 5e-4)] *= 0.99
km_kf_range = np.array([km_kf - km_kf_std, km_kf + km_kf_std])


train_fill_plot_list = [
    (mass, mass_range, ax1, "Mass $[kg]$", "$M$", "$(kg)$", (0, 0)),
    (
        ixx,
        ixx_range,
        ax2,
        "Inertia XX (YY) $[kg \ m^2]$",
        "$I_{xx} and I_{yy}$",
        "$(kg \ m^2)$",
        (-2, -2),
    ),
    (
        izz,
        izz_range,
        ax3,
        "Inertia ZZ $[kg \ m^2]$",
        "$I_{zz}$",
        "$(kg \ m^2)$",
        (-2, -2),
    ),
    (
        km_kf,
        km_kf_range,
        ax4,
        "Propellor Constant $[m]$",
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
        color=colours["train"],
        label="Train Range",
        # hatch="X",
    )

    # ax.fill_between(
    #     arm_lengths,
    #     data_range[0],
    #     data_range[1],
    #     alpha=0.35,
    #     color=colours["eval"],
    # )

    # ax.plot(arm_lengths, data, color="royalblue")
    ax.grid(linestyle="--")
    # ax.set_ylabel(f"{label} {unit}", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.ticklabel_format(axis="y", style="sci", scilimits=scilimit)
    # ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

# ## --------------- EVAL RANGE PLOT ----------------- ##


_sc_lengths = cfg.environment.scale_lengths
arm_lengths = np.linspace(_sc_lengths[1], _sc_lengths[1] + 0.06, 100)

mass = np.polyval(cfg.scale.avg_mass_fit, arm_lengths)
mass_std = np.polyval(cfg.scale.std_mass_fit, arm_lengths)
mass_range = np.array([mass - mass_std, mass + mass_std])
mass_range[mass_range < 0] = 0

ixx = np.polyval(cfg.scale.avg_ixx_fit, arm_lengths)
ixx_std = np.polyval(cfg.scale.std_ixx_fit, arm_lengths)
ixx_range = np.array([ixx - ixx_std, ixx + ixx_std])
ixx_range[ixx_range < 0] = 0

iyy = np.polyval(cfg.scale.avg_iyy_fit, arm_lengths)
iyy_std = np.polyval(cfg.scale.std_iyy_fit, arm_lengths)
iyy_range = np.array([iyy - iyy_std, iyy + iyy_std])
iyy_range[iyy_range < 0] = 0

izz = np.polyval(cfg.scale.avg_izz_fit, arm_lengths)
izz_std = np.polyval(cfg.scale.std_izz_fit, arm_lengths)
izz_range = np.array([izz - izz_std, izz + izz_std])
izz_range[izz_range < 0] = 0

km_kf = np.polyval(cfg.scale.avg_km_kf_fit, arm_lengths)
km_kf_std = np.polyval(cfg.scale.std_km_kf_fit, arm_lengths)
while np.any(km_kf - km_kf_std < 5e-4):
    km_kf_std[np.where(km_kf - km_kf_std < 5e-4)] *= 0.99
km_kf_range = np.array([km_kf - km_kf_std, km_kf + km_kf_std])


eval_fill_plot_list = [
    (mass, mass_range, ax1, "Mass", "Mass", "(kg)"),
    (ixx, ixx_range, ax2, "Inertia XX", "I_xx", "(kg m^2)"),
    # (iyy, iyy_range, ax3, "Inertia YY", "I_yy", "(kg m^2)"),
    (izz, izz_range, ax3, "Inertia ZZ", "I_zz", "(kg m^2)"),
    (km_kf, km_kf_range, ax4, "Propellor Constant", "K_m/K_f", "(m)"),
]

for i, (data, data_range, ax, title, label, unit) in enumerate(eval_fill_plot_list):
    ax.fill_between(
        arm_lengths,
        data_range[0],
        data_range[1],
        alpha=0.9,
        color=colours["eval"],
        # hatch="X",
        # edgecolor=colours["border_eval"],
        linewidth=1,
        label="Eval Range",
    )
    # ax.plot(arm_lengths, data, color="royalblue")
    ax.tick_params(axis="x", labelsize=8)


##### --------------- EVAL BORDER PLOT ----------------- ##

_sc_lengths = cfg.environment.scale_lengths
arm_lengths = np.linspace(_sc_lengths[0], _sc_lengths[1] + 0.06, 100)

mass = np.polyval(cfg.scale.avg_mass_fit, arm_lengths)
mass_std = np.polyval(cfg.scale.std_mass_fit, arm_lengths)
mass_range = np.array([mass - mass_std, mass + mass_std])
mass_range[mass_range < 0] = 0

ixx = np.polyval(cfg.scale.avg_ixx_fit, arm_lengths)
ixx_std = np.polyval(cfg.scale.std_ixx_fit, arm_lengths)
ixx_range = np.array([ixx - ixx_std, ixx + ixx_std])
ixx_range[ixx_range < 0] = 0

iyy = np.polyval(cfg.scale.avg_iyy_fit, arm_lengths)
iyy_std = np.polyval(cfg.scale.std_iyy_fit, arm_lengths)
iyy_range = np.array([iyy - iyy_std, iyy + iyy_std])
iyy_range[iyy_range < 0] = 0

izz = np.polyval(cfg.scale.avg_izz_fit, arm_lengths)
izz_std = np.polyval(cfg.scale.std_izz_fit, arm_lengths)
izz_range = np.array([izz - izz_std, izz + izz_std])
izz_range[izz_range < 0] = 0

km_kf = np.polyval(cfg.scale.avg_km_kf_fit, arm_lengths)
km_kf_std = np.polyval(cfg.scale.std_km_kf_fit, arm_lengths)
while np.any(km_kf - km_kf_std < 5e-4):
    km_kf_std[np.where(km_kf - km_kf_std < 5e-4)] *= 0.99
km_kf_range = np.array([km_kf - km_kf_std, km_kf + km_kf_std])


eval_fill_plot_list = [
    (mass, mass_range, ax1, "Mass", "Mass", "(kg)"),
    (ixx, ixx_range, ax2, "Inertia XX", "I_xx", "(kg m^2)"),
    # (iyy, iyy_range, ax3, "Inertia YY", "I_yy", "(kg m^2)"),
    (izz, izz_range, ax3, "Inertia ZZ", "I_zz", "(kg m^2)"),
    (km_kf, km_kf_range, ax4, "Propellor Constant", "K_m/K_f", "(m)"),
]

for i, (data, data_range, ax, title, label, unit) in enumerate(eval_fill_plot_list):
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

# ax5.tick_params(axis="x", labelsize=8)
# ax5.set_xlabel("Arm Length (m)")
# ax6.axis("off")

ax3.set_xlabel("Arm Length (m)")
ax4.set_xlabel("Arm Length (m)")
# legend = ax4.get_legend_handles_labels()
from matplotlib.patches import Patch

legend_handles = [
    Patch(facecolor=colours["train"], label="Train Range"),
    Patch(
        facecolor=colours["eval"],
        edgecolor=colours["border_eval"],
        linewidth=1,
        label="Eval Range",
    ),
]

# print(legend[0][0])
# legend[0][1].set(edgecolor=colours["border_eval"], linewidth=1)
# legend[1].set(eedgecolor=colours["border_eval"], linewidth=1)
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
    handles=legend_handles,
    loc="lower left",
    fontsize=10,
    fancybox=True,
    shadow=True,
    # orientation="horizontal",
    bbox_to_anchor=(-0.725, -0.325),
    ncols=2,
)

# plt.tight_layout()
# plt.show()

plt.savefig(SAVE_PATH + "changing_dynamics.svg", bbox_inches="tight")
plt.savefig(SAVE_PATH + "changing_dynamics.png", bbox_inches="tight")
