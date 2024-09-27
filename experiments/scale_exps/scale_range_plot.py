import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-deep")

# latex settings for matplotlib
from dataclasses import dataclass
import tyro
from adapt_drones.cfgs.config import *
import matplotlib.gridspec as gridspec


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

_sc_lengths = cfg.environment.scale_lengths
arm_lengths = np.linspace(_sc_lengths[0], _sc_lengths[1] + 0.04, 100)

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
km_kf_range = np.array([km_kf - km_kf_std, km_kf + km_kf_std])
km_kf_range[km_kf_range < 0] = 0

gs = gridspec.GridSpec(2, 6)
gs.update(wspace=1.0, hspace=0.25)

ax1 = plt.subplot(gs[0, :2])
ax2 = plt.subplot(gs[0, 2:4])
ax3 = plt.subplot(gs[0, 4:6])
ax4 = plt.subplot(gs[1, 1:3])
ax5 = plt.subplot(gs[1, 3:5])


fill_plot_list = [
    (mass, mass_range, ax1, "Mass", "Mass", "(kg)"),
    (ixx, ixx_range, ax2, "Inertia XX", "I_xx", "(kg m^2)"),
    (iyy, iyy_range, ax3, "Inertia YY", "I_yy", "(kg m^2)"),
    (izz, izz_range, ax4, "Inertia ZZ", "I_zz", "(kg m^2)"),
    (km_kf, km_kf_range, ax5, "Propellor Constant", "K_m/K_f", "(m)"),
]

for i, (data, data_range, ax, title, label, unit) in enumerate(fill_plot_list):
    ax.fill_between(arm_lengths, data_range[0], data_range[1], alpha=0.5)
    ax.plot(arm_lengths, data)
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel("Arm Length (m)")
    ax.set_ylabel(f"{label} {unit}")
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

# ax_mass = plt.subplot(gs[0, :2])
# ax_mass.fill_between(arm_lengths, mass_range[0], mass_range[1], alpha=0.5)
# ax_mass.plot(arm_lengths, mass)
# ax_mass.grid()
# ax_mass.set_title("Mass")
# ax_mass.tick_params(axis="x", labelsize=8)
# ax_mass.tick_params(axis="y", labelsize=8)

# ax_ixx = plt.subplot(gs[0, 2:4])
# ax_ixx.fill_between(arm_lengths, ixx_range[0], ixx_range[1], alpha=0.5)

plt.show()
