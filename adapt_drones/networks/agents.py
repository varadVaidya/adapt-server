import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from adapt_drones.networks.base_policy import (
    EnvironmentalEncoder,
    Actor,
    Critic,
    TrajectoryEncoder,
)
from adapt_drones.utils.learning import layer_init


class SimpleActorCritic(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(SimpleActorCritic, self).__init__()
        base_policy_layers = [64, 64, 64]
        base_policy_input_size = np.prod(state_shape)

        self.critic = Critic(base_policy_input_size, *base_policy_layers, output_size=1)
        self.actor_mean = Actor(
            base_policy_input_size,
            *base_policy_layers,
            output_size=np.prod(action_shape)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )
