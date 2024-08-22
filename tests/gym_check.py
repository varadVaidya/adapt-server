import os

os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import tyro

import adapt_drones
import adapt_drones.envs
from adapt_drones.cfgs.config import *


@dataclass
class Args:
    env_id: str


args = tyro.cli(Args)


cfg = Config(env_id=args.env_id)
env = gym.make(args.env_id, cfg=cfg, record=False)
check_env(env.unwrapped, warn=True, skip_render_check=True)
