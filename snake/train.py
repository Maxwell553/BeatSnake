"""Train the untrained Snake network with PPO.

Rewards (by design):
  +1  eat food (the snake got longer)
  -1  die (wall, self, or starvation)
  +5  extra bonus for filling the entire board

The same weights are trained on a mix of board sizes so the policy is not
tied to one grid.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

from snake.env import CHANNELS, N_FEATURES, VIEW_SIZE, SnakeEnv, VecSnakeEnv
from snake.model import SnakeActorCritic
from snake.ppo import PPOConfig, PPOTrainer, pick_device


DEFAULT_SIZES: List[Tuple[int, int]] = [
    (6, 6),
    (6, 8),
    (7, 7),
    (8, 8),
    (8, 10),
    (9, 9),
    (10, 10),
    (10, 12),
    (12, 8),
    (12, 12),
]


def parse_sizes(text: str) -> List[Tuple[int, int]]:
    sizes = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        w, h = part.lower().split("x")
        sizes.append((int(w), int(h)))
    if not sizes:
        raise ValueError("No sizes parsed.")
    return sizes


@torch.no_grad()
def evaluate(
    model: SnakeActorCritic,
    device: torch.device,
    sizes: Sequence[Tuple[int, int]],
    games: int = 20,
    deterministic: bool = True,
) -> dict:
    model.eval()
    results = []
    for width, height in sizes:
        lengths = []
        wins = 0
        for i in range(games):
            env = SnakeEnv(width, height, seed=10_000 + width * 100 + height + i, max_idle=None)
            obs = env.reset()
            while True:
                view = torch.as_tensor(obs["view"][None], device=device)
                features = torch.as_tensor(obs["features"][None], device=device)
                action, _, _ = model.act(view, features, deterministic=deterministic)
                obs, _, done, info = env.step(int(action.item()))
                if done:
                    lengths.append(env.length)
                    wins += int(bool(info.get("won")))
                    break
        results.append(
            {
                "size": f"{width}x{height}",
                "mean_length": float(np.mean(lengths)),
                "max_length": int(np.max(lengths)),
                "win_rate": wins / games,
                "area": width * height,
            }
        )
    mean_fill = float(np.mean([r["mean_length"] / r["area"] for r in results]))
    return {"boards": results, "mean_fill": mean_fill}


def save_checkpoint(path: Path, model: SnakeActorCritic, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "meta": meta,
        "view_size": VIEW_SIZE,
        "channels": CHANNELS,
        "n_features": N_FEATURES,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a size-invariant Snake PPO agent from scratch.")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sizes", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes) if args.sizes else DEFAULT_SIZES
    device = pick_device()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = SnakeActorCritic()
    meta = {
        "trained": False,
        "total_steps": 0,
        "sizes": sizes,
        "reward": {"eat": 1.0, "die": -1.0, "win": 5.0},
    }
    save_checkpoint(save_dir / "untrained.pt", model, meta)
    print(f"Device: {device}", flush=True)
    print(f"Saved untrained weights to {save_dir / 'untrained.pt'}", flush=True)
    print(f"Training sizes: {sizes}", flush=True)

    env = VecSnakeEnv(args.n_envs, sizes, seed=args.seed)
    cfg = PPOConfig(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        total_steps=args.total_steps,
        lr=args.lr,
        sizes=sizes,
    )
    trainer = PPOTrainer(env, model, cfg, device=device)

    steps_per_update = args.n_envs * args.n_steps
    n_updates = max(1, args.total_steps // steps_per_update)
    best_fill = -1.0
    t0 = time.time()
    history = []

    for update in range(1, n_updates + 1):
        stats = trainer.collect()
        losses = trainer.update()
        global_steps = update * steps_per_update
        row = {
            "update": update,
            "steps": global_steps,
            **stats,
            **losses,
        }
        history.append(row)
        if update % 5 == 0 or update == 1:
            elapsed = time.time() - t0
            fps = global_steps / max(elapsed, 1e-6)
            print(
                f"[{update:5d}/{n_updates}] steps={global_steps:,}  "
                f"len={stats['mean_length']:.2f}  eat={stats['eats']:.0f}  "
                f"win={stats['win_rate']:.3f}  pi={losses['policy_loss']:.3f}  "
                f"v={losses['value_loss']:.3f}  ent={losses['entropy']:.3f}  "
                f"fps={fps:.0f}",
                flush=True,
            )

        if update % args.eval_every == 0 or update == n_updates:
            eval_sizes = [(6, 6), (8, 8), (10, 10), (12, 12), (7, 9), (16, 10)]
            report = evaluate(model, device, eval_sizes, games=8)
            print(f"  eval mean_fill={report['mean_fill']:.3f}  boards={report['boards']}", flush=True)
            meta = {
                "trained": True,
                "total_steps": global_steps,
                "sizes": sizes,
                "eval": report,
                "reward": {"eat": 1.0, "die": -1.0, "win": 5.0},
            }
            save_checkpoint(save_dir / "checkpoint.pt", model, meta)
            if report["mean_fill"] >= best_fill:
                best_fill = report["mean_fill"]
                save_checkpoint(save_dir / "snake_bot.pt", model, meta)
                print(f"  saved new best bot (mean_fill={best_fill:.3f})", flush=True)

    (save_dir / "train_history.json").write_text(json.dumps(history[-200:], indent=2))
    print(f"Done. Best mean fill {best_fill:.3f}. Bot: {save_dir / 'snake_bot.pt'}", flush=True)


if __name__ == "__main__":
    main()
