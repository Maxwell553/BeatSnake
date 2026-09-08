"""Train the Snake network to clone the never-die policy and log wins.

Each iteration:
  1. Behavioral cloning on PerfectBot trajectories from random board sizes.
  2. Evaluate the unwrapped network AND the always-win policy on a fresh mix
     of board sizes (arbitrary rectangles, not one fixed grid).
  3. Append (iteration, wins) to models/win_curve.json.

The always-win policy is the covering-cycle solver; it should win every game.
The network is trained to imitate it. Watch/play use the always-win guard so a
disagreement cannot produce a death.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from snake.bot import SnakeBot
from snake.env import SnakeEnv
from snake.model import SnakeActorCritic
from snake.perfect import PerfectBot
from snake.ppo import pick_device
from snake.train import save_checkpoint

SIZES: List[Tuple[int, int]] = [
    (5, 5),
    (6, 6),
    (6, 8),
    (7, 7),
    (7, 9),
    (8, 6),
    (8, 8),
    (9, 9),
    (10, 8),
    (10, 10),
    (11, 11),
    (12, 7),
    (12, 12),
    (14, 8),
    (15, 10),
]


def sample_size(rng: np.random.Generator) -> Tuple[int, int]:
    return SIZES[int(rng.integers(0, len(SIZES)))]


def collect_demos(
    n_steps: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    views = []
    feats = []
    actions = []
    teacher = PerfectBot()
    env = SnakeEnv(6, 6, max_idle=0, seed=int(rng.integers(0, 10_000_000)))
    w, h = sample_size(rng)
    env.reset(width=w, height=h)
    while len(actions) < n_steps:
        if env.done:
            w, h = sample_size(rng)
            env.reset(width=w, height=h)
        obs = env.observe()
        action = teacher.act(env)
        views.append(obs["view"])
        feats.append(obs["features"])
        actions.append(action)
        env.step(action)
    return (
        np.stack(views),
        np.stack(feats),
        np.asarray(actions, dtype=np.int64),
    )


def play_games(policy, games: int, rng: np.random.Generator, step_limit_mult: int = 80) -> dict:
    wins = 0
    sizes = []
    fills = []
    for i in range(games):
        w, h = sample_size(rng)
        sizes.append(f"{w}x{h}")
        env = SnakeEnv(w, h, max_idle=0, seed=int(rng.integers(0, 10_000_000)))
        env.reset()
        limit = w * h * step_limit_mult
        steps = 0
        while not env.done and steps < limit:
            env.step(policy(env))
            steps += 1
        wins += int(env.won)
        fills.append(env.length / env.area)
    return {
        "wins": wins,
        "games": games,
        "sizes": sizes,
        "mean_fill": float(np.mean(fills)) if fills else 0.0,
    }


@torch.no_grad()
def neural_policy(model: SnakeActorCritic, device: torch.device):
    def act(env: SnakeEnv) -> int:
        obs = env.observe()
        view = torch.as_tensor(obs["view"][None], device=device)
        features = torch.as_tensor(obs["features"][None], device=device)
        action, _, _ = model.act(view, features, deterministic=True)
        return int(action.item())

    return act


def main() -> None:
    parser = argparse.ArgumentParser(description="Train until mixed-size wins are logged.")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--demo-steps", type=int, default=4096)
    parser.add_argument("--eval-games", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    device = pick_device()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = SnakeActorCritic().to(device)
    ckpt = save_dir / "snake_bot.pt"
    curve_path = save_dir / "win_curve.json"
    start_iter = 0
    curve = []
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
        model.load_state_dict(state)
        print(f"Resumed weights from {ckpt}", flush=True)
    if curve_path.exists():
        curve = json.loads(curve_path.read_text()).get("points", [])
        if curve:
            start_iter = int(curve[-1]["iteration"]) + 1
            print(f"Resumed log at iteration {start_iter}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    teacher = PerfectBot()
    t0 = time.time()
    if curve:
        t0 -= float(curve[-1].get("seconds", 0))

    print(f"Device {device}. Cloning never-die policy on random board sizes.", flush=True)

    for iteration in range(start_iter, args.iterations + 1):
        if iteration > 0:
            views, feats, actions = collect_demos(args.demo_steps, rng)
            model.train()
            idx = rng.permutation(len(actions))
            total_loss = 0.0
            n_mb = 0
            for start in range(0, len(actions), 512):
                mb = idx[start : start + 512]
                view = torch.as_tensor(views[mb], device=device)
                feat = torch.as_tensor(feats[mb], device=device)
                act = torch.as_tensor(actions[mb], device=device)
                dist, _ = model.forward(view, feat)
                loss = F.cross_entropy(dist.logits, act)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                n_mb += 1
            clone_loss = total_loss / max(n_mb, 1)
        else:
            clone_loss = None

        always = play_games(teacher.act, args.eval_games, rng)
        neural = play_games(neural_policy(model, device), args.eval_games, rng)
        row = {
            "iteration": iteration,
            "always_win_wins": always["wins"],
            "neural_wins": neural["wins"],
            "games": args.eval_games,
            "always_win_rate": always["wins"] / args.eval_games,
            "neural_win_rate": neural["wins"] / args.eval_games,
            "always_win_fill": always["mean_fill"],
            "neural_fill": neural["mean_fill"],
            "sizes": sorted(set(always["sizes"] + neural["sizes"])),
            "clone_loss": clone_loss,
            "seconds": time.time() - t0,
        }
        curve.append(row)
        print(
            f"iter {iteration:3d}/{args.iterations}  "
            f"always-win {always['wins']}/{args.eval_games}  "
            f"neural {neural['wins']}/{args.eval_games} fill={neural['mean_fill']:.2f}  "
            f"loss={clone_loss if clone_loss is not None else '—'}  "
            f"sizes={row['sizes']}",
            flush=True,
        )
        save_checkpoint(
            save_dir / "snake_bot.pt",
            model,
            {"trained": True, "win_curve": curve[-1], "always_win": True},
        )
        (save_dir / "win_curve.json").write_text(json.dumps({"points": curve}, indent=2))
        if (
            iteration >= 8
            and neural["wins"] == args.eval_games
            and always["wins"] == args.eval_games
        ):
            print("Neural clone matched always-win on the mixed-size suite.", flush=True)
            break

    wrapped = SnakeBot(mode="neural")
    check = play_games(wrapped.act, 16, rng)
    print(
        f"Guarded neural (deployed watcher): {check['wins']}/{check['games']} wins  sizes={check['sizes']}",
        flush=True,
    )
    print(f"Wrote {save_dir / 'win_curve.json'}", flush=True)


if __name__ == "__main__":
    main()
