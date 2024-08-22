from gymnasium.envs.registration import register

register(
    id="hover_v0",
    entry_point="adapt_drones.envs:HoverAviaryv0",
    max_episode_steps=600,
    kwargs={
        "mj_freq": 100,
        "ctrl_freq": 100,
    },
)

register(
    id="hover_v1",
    entry_point="adapt_drones.envs:HoverAviaryv1",
    max_episode_steps=600,
    kwargs={
        "mj_freq": 100,
        "ctrl_freq": 100,
    },
)
