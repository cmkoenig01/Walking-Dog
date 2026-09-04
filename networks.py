"""Actor and critic networks.

Both are small MLPs. Two choices in here are load-bearing and were not obvious
to me at the start:

LayerNorm between layers. The observation vector mixes quantities on wildly
different scales — a quaternion component sits in [-1, 1] while a joint
velocity can hit 20 rad/s. Without normalisation the large-magnitude inputs
dominate the first layer and training stalls.

A near-zero output layer. The actor's last layer is initialised with orthogonal
weights at gain 0.01 and zero bias, so the initial policy outputs approximately
zero for every joint. Since an action is an *offset* from the home pose, that
means an untrained policy starts by standing still rather than by flailing —
which is what stops the robot falling over before it has learned anything.
"""

import torch.nn as nn


class Actor(nn.Module):
    """Maps a state to a bounded mean action, one value per actuator.

    The Tanh output bounds the action to [-1, 1], which `a1_env.apply_action`
    scales by ACTION_SCALE into a joint-angle offset. Exploration noise is not
    produced here — the PPO agent owns a separate learnable log_std.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        # self.net[-2] is the final Linear, before the output Tanh.
        nn.init.orthogonal_(self.net[-2].weight, gain=0.01)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """Estimates expected discounted return from a state.

    Wider than the actor: the value function has to model the whole reward
    landscape, including terms the policy cannot directly influence, so it
    benefits from more capacity than the policy needs.
    """

    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)
