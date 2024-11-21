import numpy as np

## quaternion utils for rotation. NOTE: Keeping all of the functions in the file different from the original code
## This tbh doesn't matter much, but i want to keep the emulation code independent of the original code


def normalise_quat(q):
    return q / np.linalg.norm(q)


def conj_quat(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def vec_to_quat(v):
    """Converts a 3D vector to a quaternion."""
    q = np.zeros(4)
    q[1:] = v
    return q


def q_mult(q1, q2):
    a = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    b = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    c = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    d = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

    return np.array([a, b, c, d])


def rotate(q, v):
    """Rotate vector v by quaternion q."""
    q = normalise_quat(q)
    v = vec_to_quat(v)
    return q_mult(q, q_mult(v, conj_quat(q)))[1:]


def q_exp(q):
    """Exponential map of a quaternion."""
    a = q[0]
    v = q[1:]

    v_norm = np.linalg.norm(v)
    e_a = np.exp(a)

    exp_q = np.zeros(4)
    exp_q[0] = e_a * np.cos(v_norm)

    if v_norm > 1e-6:
        exp_q[1:] = e_a * np.sin(v_norm) * v / v_norm

    return exp_q


def sub_quat(qa, qb):
    qb = conj_quat(qb)
    qdiff = q_mult(qb, qa)
    axis = qdiff[1:]
    sin_a_2 = np.linalg.norm(axis)
    speed = 2 * np.arctan2(sin_a_2, qdiff[0])
    speed = speed if speed < np.pi else speed - 2 * np.pi

    return speed * axis


def skew_symmetric(v):

    return np.array(
        [
            [0, -v[0], -v[1], -v[2]],
            [v[0], 0, v[2], -v[1]],
            [v[1], -v[2], 0, v[0]],
            [v[2], v[1], -v[0], 0],
        ]
    )


def q_integrate(q, v, dt):
    """
    Integrate a quaternion for a timestep dt given angular velocity v
    """
    return q_mult(q_exp(vec_to_quat(v * dt / 2)), q)


class State:
    """Class that stores the state of a UAV as used in the simulator interface."""

    def __init__(
        self,
        pos=np.zeros(3),
        vel=np.zeros(3),
        quat=np.array([1, 0, 0, 0]),
        omega=np.zeros(3),
    ):
        # internally use one numpy array
        self._state = np.empty(13)
        self.pos = pos
        self.vel = vel
        self.quat = quat
        self.omega = omega

    @property
    def pos(self):
        """Position [m; world frame]."""
        return self._state[0:3]

    @pos.setter
    def pos(self, value):
        self._state[0:3] = value

    @property
    def vel(self):
        """Velocity [m/s; world frame]."""
        return self._state[3:6]

    @vel.setter
    def vel(self, value):
        self._state[3:6] = value

    @property
    def quat(self):
        """Quaternion [qw, qx, qy, qz; body -> world]."""
        return self._state[6:10]

    @quat.setter
    def quat(self, value):
        self._state[6:10] = value

    @property
    def omega(self):
        """Angular velocity [rad/s; body frame]."""
        return self._state[10:13]

    @omega.setter
    def omega(self, value):
        self._state[10:13] = value

    def __repr__(self) -> str:
        return "State pos={}, vel={}, quat={}, omega={}".format(
            self.pos, self.vel, self.quat, self.omega
        )


class Quadrotor:
    """
    Basic rigid quadrotor model for crazyflie emulation
    """

    def __init__(self, state: State):
        self.mass = 0.792
        self.J = np.array([0.0047, 0.005, 0.0074])
        self.g = np.array([0, 0, -9.81])
        self.arm_length = 0.16
        self.prop_const = 0.014

        self.inv_J = 1 / self.J

        self.state: State = state

    def step(self, u, dt):
        f_u = np.array([0, 0, u[0]])
        tau_u = -u[1:]

        vel_next = (
            self.state.vel + (self.g + rotate(self.state.quat, f_u) / self.mass) * dt
        )
        self.state.vel = vel_next

        pos_next = self.state.pos + self.state.vel * dt
        self.state.pos = pos_next

        omega_next = (
            self.state.omega
            + self.inv_J
            * (np.cross(self.J * self.state.omega, self.state.omega) + tau_u)
            * dt
        )
        self.state.omega = omega_next

        omega_global = rotate(self.state.quat, self.state.omega)
        q_next = q_integrate(self.state.quat, omega_global, dt)
        q_next = normalise_quat(q_next)
        self.state.quat = q_next

    def f_pos(self, state):
        return state.vel

    def f_vel(self, state, u):
        f_u = np.array([0, 0, u[0]])
        return self.g + rotate(state.quat, f_u / self.mass)
