import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.sans-serif": "courier",
    }
)

plt.style.use("seaborn-v0_8-paper")
import os


def data_to_box(eval_array, traj_idx, scale_idx):
    # do the sorting here
    # print(eval_array.shape)

    for i in range(eval_array.shape[0]):
        idx_sort_eval = np.argsort(np.mean(eval_array[i, :, :, 3], axis=1))
        eval_array[i, :, :, :] = eval_array[i, idx_sort_eval, :, :]

    # eval_array = eval_array[:, idx_sort_eval, :, :]

    # # remove top 3 and bottom 3 seeds
    eval_array = eval_array[:, 3:-3, :, :]

    eval_array = eval_array[traj_idx, :, :, 3]

    return eval_array[:, :, scale_idx]


if __name__ == "__main__":
    idx = [1, 7, 2, 12, 8, 5]
    scale = np.round(np.linspace(0.05, 0.22, 16), 2)
    scale_idx = [0, 5, 10, 15]
    scale = scale[scale_idx]
    traj_name = [
        "Lemniscate XZ",
        "Warped Ellipse",
        "Random Points",
        "Transposed Parabola",
        "Extended Lemniscate",
        "Satellite Orbit",
    ]

    ground_truth_mpc = np.load(
        "experiments/mpc/results-scaled/no_noise_ground_dynamics_traj_mpc_eval.npy"
    )

    changed_mpc = np.load(
        "experiments/mpc/results-scaled/changed_dynamics_traj_mpc_eval.npy"
    )

    traj_rma = np.load(
        "experiments/eval_traj/results-scale/earthy-snowball-77/wind_traj_scale_eval.npy"
    )

    colours = ["#fde725", "#5ec962", "#21918c"]

    box_ground_truth_mpc = data_to_box(ground_truth_mpc, idx, scale_idx)
    box_changed_mpc = data_to_box(changed_mpc, idx, scale_idx)
    box_rma = data_to_box(traj_rma, idx, scale_idx)

    fig, axs = plt.subplots(
        2,
        3,
        figsize=(10, 6.5),
        sharex=True,
        sharey=True,
        subplot_kw={"aspect": "auto"},
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1],
        # tight_layout=True,
    )
    xpos = np.array([_ for _ in range(scale.shape[0])])

    ground_boxes = []
    changed_boxes = []
    rma_boxes = []

    for i in range(2):
        for j in range(3):
            ground_box = axs[i, j].boxplot(
                box_ground_truth_mpc[i * 3 + j],
                positions=xpos - 0.1,
                showfliers=False,
                widths=0.25,
            )

            traj_rma_box = axs[i, j].boxplot(
                box_rma[i * 3 + j], positions=xpos, showfliers=False, widths=0.25
            )

            changed_mpc_box = axs[i, j].boxplot(
                box_changed_mpc[i * 3 + j],
                positions=xpos + 0.1,
                showfliers=False,
                widths=0.25,
            )

            axs[i, j].set_title(traj_name[i * 3 + j], fontsize=10)
            axs[i, j].set_xticks(xpos, labels=scale)
            axs[i, j].tick_params(labelsize=8)

            axs[i, j].grid(linestyle="--")

            axs[1, j].set_xlabel("Arm Length (m)", fontsize=10)
            axs[i, 0].set_ylabel("Mean Position Error (m)", fontsize=10)

            ground_boxes.append(ground_box)
            rma_boxes.append(traj_rma_box)
            changed_boxes.append(changed_mpc_box)

    all_boxes = [ground_boxes, rma_boxes, changed_boxes]

    for i in range(3):
        for box in all_boxes[i]:
            for patch in box["medians"]:
                patch.set_color(colours[i])
                patch.set_linewidth(2.5)

    labels = ["Ground Truth MPC", "Ours", "Imperfect MPC"]

    for i in range(3):
        axs[0, 0].plot([], [], "-", linewidth=2.5, color=colours[i], label=labels[i])

    axs[0, 0].legend(
        bbox_to_anchor=(0.5, 0),
        loc="lower center",
        bbox_transform=fig.transFigure,
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    # plt.tight_layout(rect=[0, 0, 1.5, 1.5], pad=0.1, h_pad=0.1, w_pad=0.1)
    plt.show()
