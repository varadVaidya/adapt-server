import numpy as np
import random


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


def random_trajectory(seed, total_time=30, dt=0.01):

    rng = np.random.default_rng(seed=seed)
    num_waypoints = 10
    t_waypoints = np.linspace(0, total_time, num_waypoints)

    spline_durations = np.diff(t_waypoints)
    num_splines = len(spline_durations)

    waypoints_xy = rng.uniform(-3, 3, (num_waypoints, 2))
    waypoints_z = rng.uniform(1, 3, num_waypoints)

    waypoints = np.hstack((waypoints_xy, waypoints_z.reshape(-1, 1)))

    velocity_constraints = np.zeros((num_waypoints, 3))
    velocity_constraints[1:-1] = rng.uniform(-1.5, 1.5, (num_waypoints - 2, 3))

    acceleration_constraints = np.zeros((num_waypoints, 3))
    acceleration_constraints[1:-1] = rng.uniform(-0.5, 0.5, (num_waypoints - 2, 3))

    boundary_conditions = np.zeros((num_splines * 6, 3))
    coeffMatrix = np.zeros((num_splines * 6, num_splines * 6))

    for i in range(num_splines):
        T = spline_durations[i]
        idx = i * 6

        boundary_conditions[idx : idx + 6] = np.array(
            [
                waypoints[i],
                velocity_constraints[i],
                acceleration_constraints[i],
                waypoints[i + 1],
                velocity_constraints[i + 1],
                acceleration_constraints[i + 1],
            ]
        )
        coeffMatrix[idx : idx + 6, idx : idx + 6] = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0],
                [T**5, T**4, T**3, T**2, T, 1],
                [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
                [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
            ]
        )

    xTrajCoeff, yTrajCoeff, zTrajCoeff = np.linalg.solve(
        coeffMatrix, boundary_conditions
    ).T

    xVelCoeff, yVelCoeff, zVelCoeff = [], [], []

    for i in range(num_splines):
        idx = i * 6
        xVelCoeff.append(np.polyder(xTrajCoeff[idx : idx + 6]))
        yVelCoeff.append(np.polyder(yTrajCoeff[idx : idx + 6]))
        zVelCoeff.append(np.polyder(zTrajCoeff[idx : idx + 6]))

    xVelCoeff = np.array(xVelCoeff)
    yVelCoeff = np.array(yVelCoeff)
    zVelCoeff = np.array(zVelCoeff)

    t = np.linspace(0, total_time, int(total_time / dt))
    reference_trajectory = np.zeros((len(t), 6))

    for i in range(num_splines):
        T = spline_durations[i]
        t_idx = np.logical_and(t >= t_waypoints[i], t <= t_waypoints[i + 1])
        t_rel = t[t_idx] - t_waypoints[i]

        reference_trajectory[t_idx, 0:3] = np.array(
            [
                np.polyval(xTrajCoeff[i * 6 : i * 6 + 6], t_rel),
                np.polyval(yTrajCoeff[i * 6 : i * 6 + 6], t_rel),
                np.polyval(zTrajCoeff[i * 6 : i * 6 + 6], t_rel),
            ]
        ).T

        reference_trajectory[t_idx, 3:6] = np.array(
            [
                np.polyval(xVelCoeff[i], t_rel),
                np.polyval(yVelCoeff[i], t_rel),
                np.polyval(zVelCoeff[i], t_rel),
            ]
        ).T

    velocity = reference_trajectory[:, 3:]
    speed_velocity = np.linalg.norm(velocity, axis=1)

    print(f"Max speed: {np.max(speed_velocity)}")

    return t, reference_trajectory[:, :3], reference_trajectory[:, 3:]


def loop_trajectory(
    discretization_dt,
    radius,
    z,
    lin_acc,
    clockwise,
    yawing,
    v_max,
):
    """
    Creates a circular trajectory on the x-y plane that increases speed by 1m/s at every revolution.

    :param quad: Quadrotor model
    :param discretization_dt: Sampling period of the trajectory.
    :param radius: radius of loop trajectory in meters
    :param z: z position of loop plane in meters
    :param lin_acc: linear acceleration of trajectory (and successive deceleration) in m/s^2
    :param clockwise: True if the rotation will be done clockwise.
    :param yawing: True if the quadrotor yaws along the trajectory. False for 0 yaw trajectory.
    :param v_max: Maximum speed at peak velocity. Revolutions needed will be calculated automatically.
    :param map_name: Name of map to load its limits
    :param plot: Whether to plot an analysis of the planned trajectory or not.
    :return: The full 13-DoF trajectory with time and input vectors
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

    # Calculate derivative of angular acceleration (alpha_vec)
    ramp_up_alpha_dt = (
        alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / ramp_up_t * ramp_t_vec)
    )
    coasting_alpha_dt = np.zeros_like(coasting_alpha)
    transition_alpha_dt = (
        -alpha_acc
        * np.pi
        / (2 * ramp_up_t)
        * np.sin(np.pi / (2 * ramp_up_t) * transition_t_vec)
    )
    alpha_dt = np.concatenate(
        (
            ramp_up_alpha_dt,
            coasting_alpha_dt,
            transition_alpha_dt,
            coasting_alpha_dt,
            ramp_up_alpha_dt,
        )
    )

    if not clockwise:
        alpha_vec *= -1
        alpha_dt *= -1

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.sin(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_z = np.ones_like(pos_traj_x) * z

    vel_traj_x = (radius * w_vec * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = -(radius * w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]

    xref = pos_traj_x.reshape(-1)
    yref = pos_traj_y.reshape(-1)
    zref = pos_traj_z.reshape(-1)

    vxref = vel_traj_x.reshape(-1)
    vyref = vel_traj_y.reshape(-1)
    vzref = np.zeros_like(vxref)

    position_ref = np.vstack((xref, yref, zref)).T
    velocity_ref = np.vstack((vxref, vyref, vzref)).T

    return t_ref, position_ref, velocity_ref


if __name__ == "__main__":
    # t, pos, vel = lemniscate_trajectory(
    #     discretization_dt=0.01,
    #     radius=5,
    #     z=1,
    #     lin_acc=0.25,
    #     clockwise=True,
    #     yawing=False,
    #     v_max=5,
    # )
    seed = -1
    seed = seed if seed > 0 else random.randint(0, 2**32 - 1)
    print(f"Seed: {seed}")
    # t, pos, vel = random_trajectory(seed=seed)
    t, pos, vel = loop_trajectory(
        discretization_dt=0.01,
        radius=5,
        z=1,
        lin_acc=0.20,
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])

    plt.show()
