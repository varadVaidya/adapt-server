import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from adapt_drones.envs.BaseAviary import BaseAviary
from adapt_drones.cfgs.config import Config
import adapt_drones.utils.rotation as rotation


class HoverAviaryv0(BaseAviary):
    """
    Simple Hover Environment. The goal is to go from
    the initial position to the final position and hover there.
    """

    # TODO: check is the action buffer is needed or not.
    def __init__(
        self,
        cfg: Config,
        init_xyz=None,
        init_quat=None,
        mj_freq: int = 100,
        ctrl_freq: int = 100,
        record: bool = False,
        camera_name: str = "trackcom",
    ):
        self.cfg: Config = cfg

        self.episode_length = cfg.environment.episode_length
        self.target_position = np.zeros(3)

        self.action_space = self._action_space()
        self.observaton_space = self._observation_space()

        super().__init__(
            init_xyz=init_xyz,
            init_quat=init_quat,
            mj_freq=mj_freq,
            ctrl_freq=ctrl_freq,
            record=record,
            camera_name=camera_name,
        )

    def _action_space(self):
        lower_bound = -1 * np.ones(4)
        upper_bound = np.ones(4)

        return spaces.Box(low=lower_bound, high=upper_bound, dtype=np.float32)

    def _observation_space(self):
        """
        observation vector =
        [delta_pos, delta_ori, delta_vel, delta_angular_vel] = # 12
        For this environment, since the goal is to hover at a desired position,
        the desired velocity and angular velocity are zero, with [1,0,0,0] as the
        desired orientation.
        """

        lower_bound = -np.inf * np.ones(12)
        upper_bound = np.inf * np.ones(12)

        return spaces.Box(low=lower_bound, high=upper_bound, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment, according to the gynmasium API."""

        super().reset(seed=seed, options=options)

        mujoco.mj_resetData(self.model, self.data)

        # kinematics reset
        self._kinematics_reset()
        # dynamics reset
        self._dynamics_reset()

        self.set_max_force_torque_limits()

        self.housekeeping()
        self.update_kinematic_data()

        # initial_obs = self._compute_obs()
        # initial_info = self._compute_info()

    def _kinematics_reset(self):
        """
        Resets the kinematics of the drone.
        """
        _trgt_xy = self.cfg.environment.target_pos_xy
        _trgt_z = self.cfg.environment.target_pos_z

        self.target_position = np.hstack(
            (
                self.np_random.uniform(_trgt_xy[0], _trgt_xy[1], 2),
                self.np_random.uniform(_trgt_z[0], _trgt_z[1], 1),
            )
        )

        _pos_xy = self.cfg.environment.pos_xy
        _pos_z = self.cfg.environment.pos_z

        init_pos = np.zeros(7)
        init_pos[0:3] = np.hstack(
            (
                self.np_random.uniform(_pos_xy[0], _pos_xy[1], 2),
                self.np_random.uniform(_pos_z[0], _pos_z[1], 1),
            )
        )

        _roll_pitch = self.cfg.environment.roll_pitch

        euler = np.concatenate(
            [self.np_random.uniform(_roll_pitch[0], _roll_pitch[1], 2), [0.0]]
        )
        mat = rotation.euler2mat(euler).reshape(9)
        mujoco.mju_mat2Quat(init_pos[3:7], mat)

        self.data.qpos = init_pos

        _lin_vel = self.cfg.environment.linear_vel
        _ang_vel = self.cfg.environment.angular_vel

        self.data.qvel = np.concatenate(
            [
                self.np_random.uniform(_lin_vel[0], _lin_vel[1], 3),
                self.np_random.uniform(_ang_vel[0], _ang_vel[1], 3),
            ]
        )

    def _dynamics_reset(self):
        """
        Resets the dynamics properties of the drone. Uses the scaling laws to
        reset the properties.
        """

        # arm length
        _arm_length = self.cfg.scale.scale_lengths
        self.arm_length = self.np_random.uniform(_arm_length[0], _arm_length[1])
        L = self.arm_length

        # mass
        _mass_avg = np.polyval(self.cfg.scale.avg_mass_fit, L)
        _mass_std = np.polyval(self.cfg.scale.std_mass_fit, L)
        _mass_std = 0.0 if _mass_std < 0.0 else _mass_std

        _mass = self.np_random.uniform(_mass_avg - _mass_std, _mass_avg + _mass_std)
        self.model.body_mass[self.drone_id] = _mass

        self.HOVER_THRUST = _mass * -1 * self.model.opt.gravity[2]

        # ixx
        _ixx_avg = np.polyval(self.cfg.scale.avg_ixx_fit, L)
        _ixx_std = np.polyval(self.cfg.scale.std_ixx_fit, L)
        _ixx_std = 0.0 if _ixx_std < 0.0 else _ixx_std

        _ixx = self.np_random.uniform(_ixx_avg - _ixx_std, _ixx_avg + _ixx_std)

        # iyy
        _iyy_avg = np.polyval(self.cfg.scale.avg_iyy_fit, L)
        _iyy_std = np.polyval(self.cfg.scale.std_iyy_fit, L)
        _iyy_std = 0.0 if _iyy_std < 0.0 else _iyy_std

        _iyy = self.np_random.uniform(_iyy_avg - _iyy_std, _iyy_avg + _iyy_std)

        # izz
        _izz_avg = np.polyval(self.cfg.scale.avg_izz_fit, L)
        _izz_std = np.polyval(self.cfg.scale.std_izz_fit, L)
        _izz_std = 0.0 if _izz_std < 0.0 else _izz_std

        _izz = self.np_random.uniform(_izz_avg - _izz_std, _izz_avg + _izz_std)

        inertia = np.array([_ixx, _iyy, _izz]).reshape(3)

        # com offset 5% of the arm length in xy and 2.5% in z
        com_offset = 0.05 * L
        com_xy = self.np_random.uniform(-com_offset, com_offset, 2)
        com_offset = 0.025 * L
        com_z = self.np_random.uniform(-com_offset, com_offset, 1)

        com = np.hstack((com_xy, com_z))
        self.model.body_ipos = com  # drone com
        self.model.site_pos[self.com_site_id] = com  # thrust com

        # km_kf
        _km_kf_avg = np.polyval(self.cfg.scale.avg_km_kf_fit, L)
        _km_kf_std = np.polyval(self.cfg.scale.std_km_kf_fit, L)
        _km_kf_std = 0.0 if _km_kf_std < 0.0 else _km_kf_std

        _km_kf = self.np_random.uniform(
            _km_kf_avg - _km_kf_std, _km_kf_avg + _km_kf_std
        )
        self.prop_const = _km_kf

        self.thrust2weight = 2.75

        # TODO: add wind.

    def housekeeping(self):
        """
        Adds more functionality to the housekeeping method in the BaseAviary class.
        """
        super().housekeeping()
        self.info_pos_error = 0.0
        self.info_vel_error = 0.0
        self.info_reward = {
            "distance_reward": 0.0,
            "velocity_reward": 0.0,
        }


if __name__ == "__main__":
    cfg = Config()
    env = HoverAviaryv0(cfg)
    env.reset()
    # env.step(env.action_space.sample())
    # env.close()
