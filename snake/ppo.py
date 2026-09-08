"""PPO trainer for the size-invariant Snake agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from snake.env import VecSnakeEnv
from snake.model import SnakeActorCritic


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class PPOConfig:
    n_envs: int = 64
    n_steps: int = 128
    total_steps: int = 2_000_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 1024
    max_grad_norm: float = 0.5
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    sizes: Optional[List[tuple]] = None


class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, view_shape, feat_dim: int) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.views = np.zeros((n_steps, n_envs, *view_shape), dtype=np.float32)
        self.features = np.zeros((n_steps, n_envs, feat_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.logprobs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.idx = 0

    def add(self, view, features, action, logprob, reward, done, value) -> None:
        self.views[self.idx] = view
        self.features[self.idx] = features
        self.actions[self.idx] = action
        self.logprobs[self.idx] = logprob
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.values[self.idx] = value
        self.idx += 1

    def compute_gae(self, last_value: np.ndarray, gamma: float, lam: float) -> None:
        last_gae = np.zeros((self.n_envs,), dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            # dones[t] is True when the episode ended after the action at t.
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values
        self.idx = 0

    def minibatches(self, batch_size: int, rng: np.random.Generator):
        n = self.n_steps * self.n_envs
        views = self.views.reshape(n, *self.views.shape[2:])
        features = self.features.reshape(n, -1)
        actions = self.actions.reshape(n)
        logprobs = self.logprobs.reshape(n)
        advantages = self.advantages.reshape(n)
        returns = self.returns.reshape(n)
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            mb = indices[start : start + batch_size]
            yield {
                "view": views[mb],
                "features": features[mb],
                "actions": actions[mb],
                "logprobs": logprobs[mb],
                "advantages": advantages[mb],
                "returns": returns[mb],
            }


class PPOTrainer:
    def __init__(
        self,
        env: VecSnakeEnv,
        model: SnakeActorCritic,
        config: PPOConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.env = env
        self.model = model
        self.cfg = config
        self.device = device or pick_device()
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
        self.rng = np.random.default_rng(0)
        self.obs = self.env.reset()
        view_shape = self.obs["view"].shape[1:]
        self.buffer = RolloutBuffer(config.n_steps, env.num_envs, view_shape, self.obs["features"].shape[-1])

    def _tensor(self, obs: Dict[str, np.ndarray]):
        view = torch.as_tensor(obs["view"], device=self.device)
        features = torch.as_tensor(obs["features"], device=self.device)
        return view, features

    @torch.no_grad()
    def collect(self) -> Dict[str, float]:
        self.model.eval()
        self.buffer.idx = 0
        ep_lens: List[float] = []
        ep_wins: List[float] = []
        eats = 0
        deaths = 0
        for _ in range(self.cfg.n_steps):
            view, features = self._tensor(self.obs)
            action, log_prob, value = self.model.act(view, features, deterministic=False)
            actions = action.cpu().numpy()
            next_obs, rewards, dones, infos = self.env.step(actions)
            self.buffer.add(
                self.obs["view"],
                self.obs["features"],
                actions,
                log_prob.cpu().numpy(),
                rewards,
                dones.astype(np.float32),
                value.cpu().numpy(),
            )
            for info, done in zip(infos, dones):
                if info.get("ate"):
                    eats += 1
                if done:
                    deaths += 1
                    ep_lens.append(float(info.get("terminal_length", 0)))
                    ep_wins.append(1.0 if info.get("won") else 0.0)
            self.obs = next_obs

        view, features = self._tensor(self.obs)
        _, _, last_value = self.model.act(view, features, deterministic=False)
        self.buffer.compute_gae(last_value.cpu().numpy(), self.cfg.gamma, self.cfg.gae_lambda)
        mean_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        win_rate = float(np.mean(ep_wins)) if ep_wins else 0.0
        return {
            "mean_length": mean_len,
            "win_rate": win_rate,
            "eats": float(eats),
            "episodes": float(len(ep_lens)),
            "deaths": float(deaths),
        }

    def update(self) -> Dict[str, float]:
        self.model.train()
        total_pi = 0.0
        total_v = 0.0
        total_ent = 0.0
        n_mb = 0
        for _ in range(self.cfg.epochs):
            for mb in self.buffer.minibatches(self.cfg.minibatch_size, self.rng):
                view = torch.as_tensor(mb["view"], device=self.device)
                features = torch.as_tensor(mb["features"], device=self.device)
                actions = torch.as_tensor(mb["actions"], device=self.device)
                old_log = torch.as_tensor(mb["logprobs"], device=self.device)
                adv = torch.as_tensor(mb["advantages"], device=self.device)
                ret = torch.as_tensor(mb["returns"], device=self.device)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                log_prob, value, entropy = self.model.evaluate(view, features, actions)
                ratio = torch.exp(log_prob - old_log)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (ret - value).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                total_pi += policy_loss.item()
                total_v += value_loss.item()
                total_ent += entropy_loss.item()
                n_mb += 1
        return {
            "policy_loss": total_pi / max(n_mb, 1),
            "value_loss": total_v / max(n_mb, 1),
            "entropy": total_ent / max(n_mb, 1),
        }
