import os

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import adapt_drones
import adapt_drones.envs
from adapt_drones.cfgs.config import *

cfg = Config()
env = gym.make("hover_v0", cfg=cfg, record=False)
check_env(env.unwrapped, warn=True, skip_render_check=True)
