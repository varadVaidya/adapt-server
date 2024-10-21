import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.sans-serif": "courier",
    }
)
plt.style.use("seaborn-v0_8-paper")


def get_npys(load_path):
    npys = []
    for root, dirs, files in os.walk(load_path):
        for file in files:
            if file.endswith(".npy"):
                npys.append(os.path.join(root, file))
    return npys


def sort_npys_speed(npys):
    print("Number of npys:", len(npys))
    fspeeds = []
    for npy in npys:
        fname = npy.split("/")[-1]
        fspeed = fname.split("_")[1]
        fspeeds.append(float(fspeed))

    fspeeds = np.array(fspeeds)
    idx_sort = np.argsort(fspeeds)
    return np.array(npys)[idx_sort].tolist()


if __name__ == "__main__":
    # Load the data
    SAVE_PATH = "/home/varad/robotics/adapt-figs/"

    load_path = "experiments/max_wind/results-wind/earthy-snowball-77"
    # npys = get_npys(load_path)
    # print(sort_npys_speed(npys))
    npys = sort_npys_speed(get_npys(load_path))
    print(npys)

    # load all the npys
    wind_data = []
    for npy in npys:
        wind_data.append(np.load(npy))

    wind_data = np.array(wind_data)

    print(wind_data.shape)

    data_compile = np.zeros(
        (wind_data.shape[1], wind_data.shape[0], wind_data.shape[3])
    )
    print(data_compile.shape)
    # data compile is a 3D array of shape (num_trajs, num_speeds, num_lengths),

    for traj_idx in range(wind_data.shape[1]):
        wind_traj_data = wind_data[:, traj_idx, :, :, 3]
        # print(wind_traj_data.shape)

        for i in range(wind_traj_data.shape[0]):
            # sort the data by the mean error
            idx_sort_eval = np.argsort(np.mean(wind_traj_data[i, :, :], axis=1))
            # print(idx_sort_eval.shape)
            wind_traj_data[i, :, :] = wind_traj_data[i, idx_sort_eval, :]

        # remove top 3 and bottom 3 seeds
        wind_traj_data = wind_traj_data[:, 3:-3, :]
        # print(wind_traj_data.shape)

        # print(np.mean(wind_traj_data, axis=1).shape)

        data_compile[traj_idx, :, :] = np.mean(wind_traj_data, axis=1)

    # traj_speed_eval = np.load(
    #     "experiments/max_speed/results-speed/earthy-snowball-77/wind_traj_speed_rma.npy"
    # )

    # for speeds in range(traj_speed_eval.shape[0]):
    #     idx_sort_eval = np.argsort(np.mean(traj_speed_eval[speeds, :, :, 3], axis=1))
    #     traj_speed_eval[speeds, :, :, :] = traj_speed_eval[speeds, idx_sort_eval, :, :]

    # # remove top 3 and bottom 3 seeds
    # traj_speed_eval = traj_speed_eval[:, 3:-3, :, :]

    # print(traj_speed_eval.shape)

    # max_speeds = traj_speed_eval[:10, 0, 0, 0]

    # data_compile = np.zeros((max_speeds.shape[0], traj_speed_eval.shape[2]))
    # print(data_compile.shape)
    # # data compile is a 2D array of shape (num_speeds, num_lengths),
    # # containing the mean error for each speed and length

    # print(traj_speed_eval[0, 0, 0, 0])
    # print(np.mean(traj_speed_eval[0, :, :, 3], axis=0).shape)
    # print(np.mean(traj_speed_eval[0, :, :, 3], axis=0))

    # for i in range(max_speeds.shape[0]):
    #     data_compile[i, :] = np.mean(traj_speed_eval[i, :, :, 3], axis=0)

    # data_compile = data_compile.T

    wind_speeds = np.arange(3, 17, step=2)
    print(wind_speeds)

    fig = plt.figure(figsize=(10, 5))
    axs = fig.subplots(1, 2, sharey=True)

    scale_idx = [0, 5, 10, 14]

    colours = ["k"] * 16
    alphas = [0.1] * 16
    line_widths = [0.5] * 16
    labels = [""] * 16
    scale_colours = ["#b5de2b", "#35b779", "#26828e", "#482878"]
    traj_titles = ["Random Points", "Satellite Orbit"]

    # replace the colours with scale_colours at the scale_idx
    for idx in scale_idx:
        colours[idx] = scale_colours[scale_idx.index(idx)]
        alphas[idx] = 1.0
        line_widths[idx] = 1.0
        labels[idx] = f"Arm Length {np.round(np.linspace(0.05, 0.22, 16)[idx], 2)} m"

    # print(colours)

    for i in range(data_compile.shape[0]):
        for j in range(data_compile.shape[2]):
            axs[i].plot(
                wind_speeds,
                data_compile[i, :, j],
                color=colours[j],
                alpha=alphas[j],
                label=labels[j],
            )

            axs[i].set_ylim([0, 0.20])
            axs[i].grid(linestyle="--")
            axs[i].set_title(traj_titles[i])
            # aspect ration to auto
            axs[i].set_xlabel("Max Wind Speed (m/s)")
            axs[i].set_aspect("auto")

    axs[0].legend()
    axs[0].set_ylabel("Mean Error (m)")

    # for i in range(data_compile.shape[0]):
    #     axs.plot(
    #         max_speeds,
    #         data_compile[i, :],
    #         color=colours[i],
    #         alpha=alphas[i],
    #         label=labels[i],
    #     )

    # axs.set_ylim([0, 0.1])
    # axs.set_xlabel("Max Speed (m/s)")
    # axs.set_ylabel("Mean Error (m)")
    # axs.grid(linestyle="--")
    # # aspect ration to auto
    # axs.set_aspect("auto")
    # axs.legend()

    plt.savefig(SAVE_PATH + "max_wind_plot.svg", bbox_inches="tight")
    plt.savefig(SAVE_PATH + "max_wind_plot.png", bbox_inches="tight")

    plt.show()
