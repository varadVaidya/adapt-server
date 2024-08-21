import os

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import adapt_drones
import adapt_drones.envs
from adapt_drones.cfgs.config import *

cfg = Config()
env = gym.make("hover_v0", cfg=cfg, record=True)
env = gym.wrappers.FlattenObservation(env)
env = gym.wrappers.RecordEpisodeStatistics(env)

obs, _ = env.reset(seed=15092024)

for _ in range(600):
    action = env.action_space.sample()
    env.step(action)

env.unwrapped.vidwrite("tests/outputs/env_check.mp4")
env.unwrapped.renderer.close()