import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-deep")
import os

if __name__ == "__main__":
    eval_file = "adapt_drones/assets/slow_pi_tcn_eval.npy"
    eval_data = np.load(eval_file, allow_pickle=True)
    print(eval_data.shape)
    idx = [1, 5, 6, 7, 3, 9]
    traj_name = [
        "Lemniscate XZ",
        "Ellipse",
        "Warped Ellipse",
        "Extended Lemniscate",
        "Circle",
        "Transposed Parabola",
    ]

    eval_data = eval_data[idx]
    print(eval_data.shape)

    fig, axs = plt.subplots(
        3,
        2,
        # figsize=(10, 20),
        layout="compressed",
        # layout="constrained",
        subplot_kw={
            "projection": "3d",
            "aspect": "equal",
            "xlim": [-2, 3],
            "ylim": [-2, 3],
            "zlim": [0, 2],
            "elev": 25,
            "azim": -60,
        },
    )
    # plt.subplots_adjust(wspace=0.25, hspace=0.1)
    max_vel = -np.inf
    for i in range(eval_data.shape[0]):
        trajectory = eval_data[i]
        rows_not_nan = sum(~np.isnan(trajectory[:, 0]))
        trajectory = trajectory[:rows_not_nan]
        max_vel = max(max_vel, np.max(np.linalg.norm(trajectory[:, 4:7], axis=1)))
    print("Max velocity:", max_vel)

    colour_bar = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, max_vel))
    colour_bar.set_array([0, max_vel])
    colour_bar.set_clim(0, max_vel)
    for i in range(eval_data.shape[0]):
        trajectory = eval_data[i]
        rows_not_nan = sum(~np.isnan(trajectory[:, 0]))
        trajectory = trajectory[:rows_not_nan]
        print("Name:", traj_name[i], "Shape:", trajectory.shape)

        scatter = axs[i % 3, i // 3].scatter(
            trajectory[:, 1],
            trajectory[:, 2],
            trajectory[:, 3],
            s=0.75,
            c=colour_bar.to_rgba(np.linalg.norm(trajectory[:, 4:7], axis=1)),
        )
        axs[i % 3, i // 3].set_title(traj_name[i])
        axs[i % 3, i // 3].set_xlabel("X(m)")
        axs[i % 3, i // 3].set_ylabel("Y(m)")
        # axs[i % 3, i//3].set_zlabel("Z(m)")
        axs[i % 3, i // 3].xaxis.set_major_locator(plt.MaxNLocator(4))
        axs[i % 3, i // 3].yaxis.set_major_locator(plt.MaxNLocator(4))
        axs[i % 3, i // 3].zaxis.set_major_locator(plt.MaxNLocator(4))
    axs[0, 1].set_zlabel("Z(m)")
    axs[1, 1].set_zlabel("Z(m)")
    axs[2, 1].set_zlabel("Z(m)")
    fig.colorbar(
        colour_bar,
        ax=axs,
        label="Velocity (m/s)",
        shrink=0.75,
        orientation="horizontal",
    )
    # plt.savefig("experiments/eval_trajs_portrait.png")
    plt.show()
