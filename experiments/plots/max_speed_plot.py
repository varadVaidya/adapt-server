import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.sans-serif": "courier",
    }
)

plt.style.use("seaborn-v0_8-paper")
if __name__ == "__main__":
    # Load the data
    SAVE_PATH = "/home/varad/robotics/adapt-figs/"

    traj_speed_eval = np.load(
        "experiments/max_speed/results-speed/earthy-snowball-77/wind_traj_speed_rma.npy"
    )

    for speeds in range(traj_speed_eval.shape[0]):
        idx_sort_eval = np.argsort(np.mean(traj_speed_eval[speeds, :, :, 3], axis=1))
        traj_speed_eval[speeds, :, :, :] = traj_speed_eval[speeds, idx_sort_eval, :, :]

    # remove top 3 and bottom 3 seeds
    traj_speed_eval = traj_speed_eval[:, 3:-3, :, :]

    print(traj_speed_eval.shape)

    max_speeds = traj_speed_eval[:10, 0, 0, 0]

    data_compile = np.zeros((max_speeds.shape[0], traj_speed_eval.shape[2]))
    print(data_compile.shape)
    # data compile is a 2D array of shape (num_speeds, num_lengths),
    # containing the mean error for each speed and length

    print(traj_speed_eval[0, 0, 0, 0])
    print(np.mean(traj_speed_eval[0, :, :, 3], axis=0).shape)
    print(np.mean(traj_speed_eval[0, :, :, 3], axis=0))

    for i in range(max_speeds.shape[0]):
        data_compile[i, :] = np.mean(traj_speed_eval[i, :, :, 3], axis=0)

    data_compile = data_compile.T

    fig = plt.figure(figsize=(5, 5))
    axs = fig.subplots(1, 1)

    scale_idx = [0, 5, 10, 14]

    colours = ["k"] * 16
    alphas = [0.1] * 16
    line_widths = [0.5] * 16
    labels = [""] * 16
    scale_colours = ["#b5de2b", "#35b779", "#26828e", "#482878"]

    # replace the colours with scale_colours at the scale_idx
    for idx in scale_idx:
        colours[idx] = scale_colours[scale_idx.index(idx)]
        alphas[idx] = 1.0
        line_widths[idx] = 1.0
        labels[idx] = f"Arm Length {np.round(np.linspace(0.05, 0.22, 16)[idx], 2)} m"

    print(colours)

    for i in range(data_compile.shape[0]):
        axs.plot(
            max_speeds,
            data_compile[i, :],
            color=colours[i],
            alpha=alphas[i],
            label=labels[i],
        )

    axs.set_ylim([0, 0.1])
    axs.set_xlabel("Max Speed (m/s)")
    axs.set_ylabel("Mean Error (m)")
    axs.grid(linestyle="--")
    # aspect ration to auto
    axs.set_aspect("auto")
    axs.legend()

    plt.savefig(SAVE_PATH + "max_speed_plot.svg", bbox_inches="tight")
    plt.savefig(SAVE_PATH + "max_speed_plot.png", bbox_inches="tight")

    plt.show()
