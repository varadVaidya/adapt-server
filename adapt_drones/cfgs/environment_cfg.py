from dataclasses import dataclass


@dataclass
class HoverAviaryv0Config:
    eval: bool

    scale: bool
    scale_lengths: list

    pos_xy: list
    pos_z: float

    target_pos_xy: list
    target_pos_z: list

    linear_vel: list
    angular_vel: list

    roll_pitch: list

    env_id: str = "hover_v0"
    episode_length: int = 6  # secs
    agent_name: str = "AC"

    def __init__(self, eval, scale):
        self.eval = eval
        self.scale = scale

        self.pos_xy = [-1.0, 1.0] if not eval else [-1.25, 1.25]
        self.pos_z = [1.0, 2.0] if not eval else [0.75, 2.25]

        self.target_pos_xy = [-1.0, 1.0] if not eval else [-1.25, 1.25]
        self.target_pos_z = [1.0, 2.0] if not eval else [0.75, 2.25]

        self.linear_vel = [-0.1, 0.1] if not eval else [-0.125, 0.125]
        self.angular_vel = [-0.05, 0.05] if not eval else [-0.05, 0.05]

        self.roll_pitch = [-0.15, 0.15] if not eval else [-0.15, 0.15]

        self.scale_lengths = [0.05, 0.16] if self.scale else [0.16, 0.16]


@dataclass
class HoverAviaryv1Config:
    eval: bool

    scale: bool
    scale_lengths: list

    pos_xy: list
    pos_z: float

    target_pos_xy: list
    target_pos_z: list

    linear_vel: list
    angular_vel: list

    roll_pitch: list

    env_id: str = "hover_v1"
    episode_length: int = 6  # secs
    agent_name: tuple = ("AC", "RMA")

    def __init__(self, eval, scale):
        self.eval = eval
        self.scale = scale

        self.pos_xy = [-1.0, 1.0] if not eval else [-1.25, 1.25]
        self.pos_z = [1.0, 2.0] if not eval else [0.75, 2.25]

        self.target_pos_xy = [-1.0, 1.0] if not eval else [-1.25, 1.25]
        self.target_pos_z = [1.0, 2.0] if not eval else [0.75, 2.25]

        self.linear_vel = [-0.1, 0.1] if not eval else [-0.125, 0.125]
        self.angular_vel = [-0.05, 0.05] if not eval else [-0.05, 0.05]

        self.roll_pitch = [-0.15, 0.15] if not eval else [-0.15, 0.15]

        self.scale_lengths = [0.05, 0.16] if self.scale else [0.16, 0.16]
