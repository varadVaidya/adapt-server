""" Implementation of the Simplified Simulator and its quadrotor dynamics. """

from math import sqrt
import numpy as np
from adapt_drones.utils.mpc_utils import (
    quaternion_to_euler,
    skew_symmetric,
    skew_symmetric,
    v_dot_q,
    unit_quat,
    quaternion_inverse,
)


class Quadrotor3D:

    def __init__(self, noisy: bool = False):
        """
        Initialize the quadrotor dynamics.

        :param noisy:
        """

        self.noisy = noisy
        # maximum thrust of the quadrotor in N
        self.max_thrust = 20

        # state space
        self.pos = np.zeros((3,))
        self.vel = np.zeros((3,))
        # quaternion format: [w, x, y, z]
        self.angle = np.array([1, 0, 0, 0])
        self.a_rate = np.zeros((3,))

        # input constraints
        self.max_input_value = 1  # full throttle
        self.min_input_value = 0  # no throttle

        # dynamics
        self.J = np.array([0.03, 0.03, 0.06])  # kg  m^2
        self.mass = 1.0  # kg

        # lenght of motor to CoG
        self.length = 0.24  # m

        # position of thrusters
        h = np.cos(np.pi / 4) * self.length
        self.x_f = np.array([h, -h, -h, h])
        self.y_f = np.array([-h, -h, h, h])

        # for z thrust toque
        self.c = 0.013  # m
        self.z_l_tau = np.array([-self.c, self.c, -self.c, self.c])

        # Gravity vector
        self.g = np.array([[0], [0], [9.81]])  # m s^-2

        # Actuation thrusts
        self.u_noiseless = np.array([0.0, 0.0, 0.0, 0.0])
        self.u = np.array([0.0, 0.0, 0.0, 0.0])  # N

    def set_state(self, *args, **kwargs):
        """
        Set the state of the quadrotor.
        """
        if len(args) != 0:
            assert len(args) == 1 and len(args[0]) == 13

            self.pos = np.array(args[0][:3])
            self.angle = np.array(args[0][3:7])
            self.vel = np.array(args[0][7:10])
            self.a_rate = np.array(args[0][10:])

        else:
            self.pos = kwargs["pos"]
            self.angle = kwargs["angle"]
            self.vel = kwargs["vel"]
            self.a_rate = kwargs["rate"]

    def get_state(self, quaternion=False, stacked=False):

        if quaternion and not stacked:
            return [self.pos, self.angle, self.vel, self.a_rate]
        if quaternion and stacked:
            return [
                self.pos[0],
                self.pos[1],
                self.pos[2],
                self.angle[0],
                self.angle[1],
                self.angle[2],
                self.angle[3],
                self.vel[0],
                self.vel[1],
                self.vel[2],
                self.a_rate[0],
                self.a_rate[1],
                self.a_rate[2],
            ]

        angle = quaternion_to_euler(self.angle)
        if not quaternion and stacked:
            return [
                self.pos[0],
                self.pos[1],
                self.pos[2],
                angle[0],
                angle[1],
                angle[2],
                self.vel[0],
                self.vel[1],
                self.vel[2],
                self.a_rate[0],
                self.a_rate[1],
                self.a_rate[2],
            ]
        return [self.pos, angle, self.vel, self.a_rate]

    def get_control(self, noisy=False):
        if not noisy:
            return self.u_noiseless
        else:
            return self.u

    def update(self, u, dt):
        """
        Runge-Kutta 4th order dynamics integration

        :param u: 4-dimensional vector with components between [0.0, 1.0] that represent the activation of each motor.
        :param dt: time differential
        """
        # Clip inputs
        for i, u_i in enumerate(u):
            self.u_noiseless[i] = max(
                min(u_i, self.max_input_value), self.min_input_value
            )
        self.u = self.u_noiseless * self.max_thrust

        # Generate disturbance forces / torques
        if self.noisy:
            f_d = np.random.normal(size=(3, 1), scale=10 * dt)
            t_d = np.random.normal(size=(3, 1), scale=10 * dt)
        else:
            f_d = np.zeros((3, 1))
            t_d = np.zeros((3, 1))

        x = self.get_state(quaternion=True, stacked=False)

        # RK4 integration
        k1 = [
            self.f_pos(x),
            self.f_att(x),
            self.f_vel(x, self.u, f_d),
            self.f_rate(x, self.u, t_d),
        ]
        x_aux = [x[i] + dt / 2 * k1[i] for i in range(4)]
        k2 = [
            self.f_pos(x_aux),
            self.f_att(x_aux),
            self.f_vel(x_aux, self.u, f_d),
            self.f_rate(x_aux, self.u, t_d),
        ]
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(4)]
        k3 = [
            self.f_pos(x_aux),
            self.f_att(x_aux),
            self.f_vel(x_aux, self.u, f_d),
            self.f_rate(x_aux, self.u, t_d),
        ]
        x_aux = [x[i] + dt * k3[i] for i in range(4)]
        k4 = [
            self.f_pos(x_aux),
            self.f_att(x_aux),
            self.f_vel(x_aux, self.u, f_d),
            self.f_rate(x_aux, self.u, t_d),
        ]
        x = [
            x[i]
            + dt
            * (
                1.0 / 6.0 * k1[i]
                + 2.0 / 6.0 * k2[i]
                + 2.0 / 6.0 * k3[i]
                + 1.0 / 6.0 * k4[i]
            )
            for i in range(4)
        ]

        # Ensure unit quaternion
        x[1] = unit_quat(x[1])

        self.pos, self.angle, self.vel, self.a_rate = x

    def f_pos(self, x):
        """
        Time-derivative of the position vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: position differential increment (vector): d[pos_x; pos_y]/dt
        """

        vel = x[2]
        return vel

    def f_att(self, x):
        """
        Time-derivative of the attitude in quaternion form
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: attitude differential increment (quaternion qw, qx, qy, qz): da/dt
        """

        rate = x[3]
        angle_quaternion = x[1]

        return 1 / 2 * skew_symmetric(rate).dot(angle_quaternion)

    def f_vel(self, x, u, f_d):
        """
        Time-derivative of the velocity vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
        :param f_d: disturbance force vector (3-dimensional)
        :return: 3D velocity differential increment (vector): d[vel_x; vel_y; vel_z]/dt
        """

        a_thrust = np.array([[0], [0], [np.sum(u)]]) / self.mass

        angle_quaternion = x[1]

        return np.squeeze(
            -self.g + v_dot_q(a_thrust + f_d / self.mass, angle_quaternion)
        )

    def f_rate(self, x, u, t_d):
        """
        Time-derivative of the angular rate
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
        :param t_d: disturbance torque (3D)
        :return: angular rate differential increment (scalar): dr/dt
        """

        rate = x[3]
        return np.array(
            [
                1
                / self.J[0]
                * (
                    u.dot(self.y_f)
                    + t_d[0]
                    + (self.J[1] - self.J[2]) * rate[1] * rate[2]
                ),
                1
                / self.J[1]
                * (
                    -u.dot(self.x_f)
                    + t_d[1]
                    + (self.J[2] - self.J[0]) * rate[2] * rate[0]
                ),
                1
                / self.J[2]
                * (
                    u.dot(self.z_l_tau)
                    + t_d[2]
                    + (self.J[0] - self.J[1]) * rate[0] * rate[1]
                ),
            ]
        ).squeeze()
