import os
import random
import time
import yaml
from dataclasses import asdict
import subprocess

import numpy as np
import torch
import gymnasium as gym

from adapt_drones.cfgs.config import *
from adapt_drones.utils.learning import make_env
from adapt_drones.networks.ppo import ppo_train
from adapt_drones.utils.git_utils import check_git_clean

check_git_clean()


@dataclass
class Args:
    env_id: str
    seed: int = 15092024


args = tyro.cli(Args)
cfg = Config(env_id=args.env_id, seed=args.seed)

# set random seeds
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic


device = torch.device(
    "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
)

if cfg.experiment.track:
    import wandb

    run = wandb.init(
        project=cfg.experiment.wandb_project_name,
        group=cfg.experiment.grp_name,
        config=asdict(cfg),
        sync_tensorboard=True,
        settings=wandb.Settings(_disable_stats=True),
    )
    print("Wandb Initialized")
    cfg.run_name = run.name
    cfg.experiment.run_name = run.name
    commit = subprocess.check_output(["git", "log", "--format=%H", "-n", "1"])
    branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    tag_name = "runs/" + cfg.experiment.grp_name + "/" + cfg.run_name
    run.notes = f"Commit: {commit.decode('utf-8')}\nBranch: {branch_name}"

    print("Tagging the commit")
    # tag the commit with the tag_name
    subprocess.run(["git", "tag", tag_name])
    print("Commit: ", commit.decode("utf-8"))
    print("Branch: ", branch_name)
    print("Tag: ", tag_name)

# env_setup
envs = gym.vector.SyncVectorEnv(
    [make_env(cfg.env_id, cfg=cfg) for _ in range(cfg.learning.num_envs)]
)

ppo_train(args=cfg, envs=envs)

with open("runs/" + cfg.grp_name + "/" + cfg.run_name + "/config.yaml", "w") as f:
    yaml.dump(asdict(cfg), f)
    f.close()
