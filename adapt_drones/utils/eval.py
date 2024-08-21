import os
import random
from dataclasses import asdict

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from adapt_drones.cfgs.config import *
from adapt_drones.networks.agents import *
from adapt_drones.utils.ploting import data_plot, TextonPlot


def phase1_eval(cfg: Config, duration: int = 6, best_model: bool = True):
    """Phase 1 evaluation script

    Args:
        cfg (Config): main config file
        duration (int, optional): Seconds of simulation to run. Defaults to 6.
        best_model (bool, optional): Use the best saved mode. Defaults to True.
        Uses the final model if False.
    """
    print("=================================")
    print("Phase 1 Evaluation")

    # seeding again for sanity
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    rng = np.random.default_rng(seed=cfg.seed)

    # load the model
    run_folder = "runs/" + cfg.grp_name + "/" + cfg.run_name + "/"
    results_folder = run_folder + "results/"
    archive_folder = run_folder + "archive/"
    datadump_folder = results_folder + "datadump/"

    os.makedirs(datadump_folder, exist_ok=True)
    os.makedirs(archive_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    model_path = (
        run_folder + "best_model.pt" if best_model else run_folder + "final_model.pt"
    )
    print("Model Path:", model_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
    )

    env = gym.make(cfg.env_id, cfg=cfg, record=True)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if cfg.agent == "AC":
        agent = SimpleActorCritic(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
        ).to(device)
        agent.load_state_dict(torch.load(model_path))

    else:
        raise ValueError("Invalid agent type")

    agent.eval()
    obs, _ = env.reset(seed=cfg.seed)

    mass = env.unwrapped.model.body_mass[env.unwrapped.drone_id]
    inertia = env.unwrapped.model.body_inertia[env.unwrapped.drone_id]
    wind = env.unwrapped.model.opt.wind
    com = env.unwrapped.model.body_ipos[env.unwrapped.drone_id]

    prop_const = env.unwrapped.prop_const
    arm_length = env.unwrapped.arm_length
    thrust2weight = env.unwrapped.thrust2weight

    text_plot = TextonPlot(
        seed=f"Seed: {cfg.seed}",
        mass=f"Mass: {mass:.3f}",
        inertia=f"Inertia: {inertia}",
        wind=f"Wind: {wind}",
        com=f"Com:{com}",
        prop_const=f"Prop Constant:{prop_const}",
        arm_length=f"Arm Length:{arm_length}",
        thrust2weight=f"TWR:{thrust2weight}",
    )

    print("\n".join("{}".format(v) for k, v in asdict(text_plot).items()))

    t, ref_positon, ref_velocity = env.unwrapped.eval_trajectory(duration=duration)

    position, velocity = [], []

    for i in range(len(t)):
        env.unwrapped.target_position = ref_positon[i]
        action = agent.get_action_and_value(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        )[0]
        obs = env.step(action.cpu().numpy()[0])[0]

        position.append(env.unwrapped.position)
        velocity.append(env.unwrapped.velocity)

    position = np.array(position)
    velocity = np.array(velocity)

    datadump = np.hstack(
        (t.reshape(-1, 1), position, velocity, ref_positon, ref_velocity)
    )
    headers = ["p", "v", "pd", "vd"]
    axes = ["x", "y", "z"]
    headers = [f"{a}_{b}" for a in headers for b in axes]
    headers = ["t"] + headers
    np.savetxt(
        datadump_folder + f"phase1-{cfg.seed}.csv",
        datadump,
        delimiter=",",
        header=",".join(headers),
    )

    data_plot(
        t,
        position=position,
        goal_position=ref_positon,
        velocity=velocity,
        goal_velocity=ref_velocity,
        plot_text=text_plot,
        save_prefix="phase_1",
        save_path=results_folder,
    )
    env.unwrapped.vidwrite(results_folder + "phase_1.mp4")
    env.unwrapped.renderer.close()
