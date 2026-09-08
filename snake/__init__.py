"""Size-invariant Snake deep-RL package."""

__version__ = "1.0.0"

from snake.env import CHANNELS, N_ACTIONS, N_FEATURES, VIEW_SIZE, SnakeEnv
from snake.model import SnakeActorCritic

__all__ = [
    "CHANNELS",
    "N_ACTIONS",
    "N_FEATURES",
    "VIEW_SIZE",
    "SnakeEnv",
    "SnakeActorCritic",
]
