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


def data_to_violin(eval_array, idx):
    # do the sorting here
    # print(eval_array.shape)

    for i in range(eval_array.shape[0]):
        idx_sort_eval = np.argsort(np.mean(eval_array[i, :, :, 3], axis=1))
        eval_array[i, :, :, :] = eval_array[i, idx_sort_eval, :, :]

    eval_array = eval_array[:, idx_sort_eval, :, :]

    # # remove top 3 and bottom 3 seeds
    eval_array = eval_array[:, 3:-3, :, :]

    print(eval_array.shape)

    return eval_array[idx, :, :, 3]


if __name__ == "__main__":
    idx = [1, 7, 2, 12, 8, 5]
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

    # print(ground_truth_mpc.shape)

    # MANIPULATE DATA INTO THE SHAPE OF (idxs, mean_error_across_seeds, scale)

    violin_ground_truth_mpc = data_to_violin(ground_truth_mpc, idx)
    violin_changed_mpc = data_to_violin(changed_mpc, idx)
    violin_rma = data_to_violin(traj_rma, idx)

    # one trajectory
    violin_ground_truth_mpc = violin_ground_truth_mpc[0]
    violin_changed_mpc = violin_changed_mpc[0]
    violin_rma = violin_rma[0]

    # print(violin_changed_mpc)

    fig, ax = plt.subplots(figsize=(6, 4))
    scale = np.round(np.linspace(0.05, 0.22, 16), 2)
    # xpos =range(violin_ground_truth_mpc.shape[1])
    xpos = np.array([_ for _ in range(violin_ground_truth_mpc.shape[1])])

    # ax.violinplot(
    #     violin_ground_truth_mpc,
    #     showmeans=True,
    #     showmedians=False,
    #     positions=xpos - 0.1,
    # )
    # ax.violinplot(
    #     violin_rma,
    #     showmeans=True,
    #     showmedians=False,
    #     positions=xpos,
    # )
    # ax.violinplot(
    #     violin_changed_mpc,
    #     showmeans=True,
    #     showmedians=False,
    #     positions=xpos + 0.1,
    # )

    ax.boxplot(
        violin_ground_truth_mpc,
        positions=xpos - 0.1,
        showfliers=False,
    )
    ax.boxplot(
        violin_rma,
        positions=xpos,
        showfliers=False,
    )
    ax.boxplot(
        violin_changed_mpc,
        positions=xpos + 0.1,
        showfliers=False,
    )

    ax.set_xticks(xpos, labels=scale)
    plt.show()

    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    # # Fixing random state for reproducibility
    # np.random.seed(19680801)

    # # generate some random test data
    # all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

    # print(type(all_data), len(all_data), len(all_data[0]))
    # all_data = np.array(all_data).T
    # print(all_data.shape)

    # # plot violin plot
    # axs[0].violinplot(all_data, showmeans=False, showmedians=True)
    # axs[0].set_title("Violin plot")

    # # plot box plot
    # axs[1].boxplot(all_data)
    # axs[1].set_title("Box plot")

    # # adding horizontal grid lines
    # for ax in axs:
    #     ax.yaxis.grid(True)
    #     ax.set_xticks(
    #         [y + 1 for y in range(all_data.shape[1])], labels=["x1", "x2", "x3", "x4"]
    #     )
    #     ax.set_xlabel("Four separate samples")
    #     ax.set_ylabel("Observed values")

    # plt.show()
