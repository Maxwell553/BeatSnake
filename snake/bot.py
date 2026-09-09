"""Packaged Snake bot: neural, perfect, or hybrid."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from snake.env import SnakeEnv
from snake.model import SnakeActorCritic
from snake.perfect import HybridBot, PerfectBot
from snake.ppo import pick_device
from snake.train import load_compatible_state


def default_model_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "models" / "snake_bot.pt"


def load_network(path: Optional[str] = None, device: Optional[torch.device] = None) -> SnakeActorCritic:
    device = device or pick_device()
    model_path = Path(path) if path else default_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained weights at {model_path}. Run `python -m snake train` first."
        )
    payload = torch.load(model_path, map_location=device, weights_only=False)
    model = SnakeActorCritic()
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    load_compatible_state(model, state)
    model.to(device)
    model.eval()
    return model


class SnakeBot:
    """Public inference API used by play / evaluate / any external wrapper."""

    def __init__(
        self,
        mode: str = "neural",
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if mode not in {"neural", "perfect", "hybrid", "random"}:
            raise ValueError("mode must be neural, perfect, hybrid, or random")
        self.mode = mode
        self.device = device or pick_device()
        self.model: Optional[SnakeActorCritic] = None
        self.perfect = PerfectBot()
        if mode in {"neural", "hybrid"}:
            self.model = load_network(model_path, self.device)
        if mode == "hybrid":
            self._hybrid = HybridBot(self._neural_action, self.perfect)

    def act(self, env: SnakeEnv) -> int:
        if self.mode == "random":
            return int(np.random.randint(0, 3))
        if self.mode == "perfect" or self.model is None:
            return self.perfect.act(env)
        if self.mode == "neural":
            # Unwrapped network only — no covering-cycle / search fallback.
            return self._neural_action(env)
        # hybrid: network proposal, PerfectBot if that move is unsafe
        return self._hybrid.act(env)

    @torch.no_grad()
    def _neural_action(self, env: SnakeEnv) -> int:
        assert self.model is not None
        obs = env.observe()
        view = torch.as_tensor(obs["view"][None], device=self.device)
        features = torch.as_tensor(obs["features"][None], device=self.device)
        action, _, _ = self.model.act(view, features, deterministic=True)
        return int(action.item())
