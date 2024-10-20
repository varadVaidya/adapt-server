import numpy as np


def lemniscate_trajectory(
    discretization_dt, radius, z, lin_acc, clockwise, yawing, v_max
):
    """

    :param quad:
    :param discretization_dt:
    :param radius:
    :param z:
    :param lin_acc:
    :param clockwise:
    :param yawing:
    :param v_max:
    :param map_name:
    :param plot:
    :return:
    """

    assert z > 0

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    ramp_t_vec = np.arange(0, ramp_up_t, discretization_dt)
    ramp_up_alpha = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * ramp_t_vec) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    coasting_t_vec = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    coasting_alpha = np.ones_like(coasting_t_vec) * alpha_acc
    # Transition phase: decelerate
    transition_t_vec = np.arange(0, 2 * ramp_up_t, discretization_dt)
    transition_alpha = alpha_acc * np.cos(np.pi / (2 * ramp_up_t) * transition_t_vec)
    transition_t_vec += coasting_t_vec[-1] + discretization_dt
    # Deceleration phase
    down_coasting_t_vec = (
        transition_t_vec[-1]
        + np.arange(0, coasting_duration, discretization_dt)
        + discretization_dt
    )
    down_coasting_alpha = -np.ones_like(down_coasting_t_vec) * alpha_acc
    # Bring to rest phase
    ramp_up_t_vec = (
        down_coasting_t_vec[-1]
        + np.arange(0, ramp_up_t, discretization_dt)
        + discretization_dt
    )
    ramp_up_alpha_end = ramp_up_alpha - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate(
        (
            ramp_t_vec,
            coasting_t_vec,
            transition_t_vec,
            down_coasting_t_vec,
            ramp_up_t_vec,
        )
    )
    alpha_vec = np.concatenate(
        (
            ramp_up_alpha,
            coasting_alpha,
            transition_alpha,
            down_coasting_alpha,
            ramp_up_alpha_end,
        )
    )

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Adaption: we achieve the highest spikes in the bodyrates when passing through the 'center' part of the figure-8
    # This leads to negative reference thrusts.
    # Let's see if we can alleviate this by adapting the z-reference in these parts to add some acceleration in the
    # z-component
    z_dim = 0.0

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = (
        radius * (np.sin(angle_vec) * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    )
    pos_traj_z = -z_dim * np.cos(4.0 * angle_vec)[np.newaxis, np.newaxis, :] + z

    vel_traj_x = -radius * (w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = (
        radius
        * (w_vec * np.cos(angle_vec) ** 2 - w_vec * np.sin(angle_vec) ** 2)[
            np.newaxis, np.newaxis, :
        ]
    )
    vel_traj_z = (
        4.0 * z_dim * w_vec * np.sin(4.0 * angle_vec)[np.newaxis, np.newaxis, :]
    )

    x_ref = pos_traj_x.reshape(-1)
    y_ref = pos_traj_y.reshape(-1)
    z_ref = pos_traj_z.reshape(-1)

    vx_ref = vel_traj_x.reshape(-1)
    vy_ref = vel_traj_y.reshape(-1)
    vz_ref = vel_traj_z.reshape(-1)

    position_ref = np.vstack((x_ref, y_ref, z_ref)).T
    velocity_ref = np.vstack((vx_ref, vy_ref, vz_ref)).T

    return t_ref, position_ref, velocity_ref


if __name__ == "__main__":
    t, pos, vel = lemniscate_trajectory(
        discretization_dt=0.01,
        radius=5,
        z=1,
        lin_acc=0.25,
        clockwise=True,
        yawing=False,
        v_max=5,
    )

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(t, pos[:, 0], label="x")
    axs[0].plot(t, pos[:, 1], label="y")
    axs[0].plot(t, pos[:, 2], label="z")

    axs[1].plot(t, vel[:, 0], label="vx")
    axs[1].plot(t, vel[:, 1], label="vy")
    axs[1].plot(t, vel[:, 2], label="vz")

    axs[0].legend()
    axs[1].legend()

    axs[2].plot(pos[:, 0], pos[:, 1])
    axs[2].set_aspect("equal")

    plt.savefig("lemniscate_trajectory.png")
