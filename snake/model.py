"""Size-invariant actor-critic for Snake.

The convolutional trunk only sees a fixed 11x11 egocentric window, so the
parameter count does not grow with board size. Global scalars (food bearing,
length / area, wall distances) are normalized by board dimensions, which is
what lets one set of weights play any grid.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from snake.env import CHANNELS, N_ACTIONS, N_FEATURES, VIEW_SIZE


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SnakeActorCritic(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # 11x11 --stride2--> 6x6
        conv_out = 64 * 6 * 6
        self.body = nn.Sequential(
            nn.Linear(conv_out + N_FEATURES, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.actor = nn.Linear(hidden, N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)
        # Soft prior: boost actions whose relative leftover-to-food feature is ~0
        # (PerfectBot's preferred safe cycle shortcut / rail step).
        self.rel_action_boost = nn.Parameter(torch.tensor(4.0))

        self.conv.apply(lambda m: orthogonal_init(m, gain=np_sqrt2()))
        self.body.apply(lambda m: orthogonal_init(m, gain=np_sqrt2()))
        orthogonal_init(self.actor, gain=0.01)
        orthogonal_init(self.critic, gain=1.0)

    def encode(self, view: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = self.conv(view)
        x = x.flatten(1)
        x = torch.cat([x, features], dim=-1)
        return self.body(x)

    def forward(
        self, view: torch.Tensor, features: torch.Tensor
    ) -> Tuple[Categorical, torch.Tensor]:
        h = self.encode(view, features)
        logits = self.actor(h)
        if features.shape[-1] >= 27:
            rel = features[:, 24:27]
            # Only boost when exactly one action is the PerfectBot leftover winner.
            pref = (rel <= 1e-5).to(logits.dtype)
            unique = (pref.sum(dim=-1, keepdim=True) == 1).to(logits.dtype)
            logits = logits + self.rel_action_boost * pref * unique
        dist = Categorical(logits=logits)
        value = self.critic(h).squeeze(-1)
        return dist, value

    def act(
        self, view: torch.Tensor, features: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(view, features)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(
        self, view: torch.Tensor, features: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(view, features)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy


def np_sqrt2() -> float:
    return 2.0 ** 0.5
