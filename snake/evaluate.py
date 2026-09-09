"""Evaluate neural / perfect / hybrid bots across many board sizes."""

from __future__ import annotations

import argparse
import json
from typing import List, Tuple

from snake.bot import SnakeBot
from snake.env import SnakeEnv


def parse_sizes(text: str) -> List[Tuple[int, int]]:
    sizes = []
    for part in text.split(","):
        part = part.strip()
        w, h = part.lower().split("x")
        sizes.append((int(w), int(h)))
    return sizes


def run_games(bot: SnakeBot, width: int, height: int, games: int, seed: int) -> dict:
    lengths = []
    wins = 0
    deaths = 0
    reasons = {}
    for i in range(games):
        idle = 0 if bot.mode in {"perfect", "hybrid"} else None
        env = SnakeEnv(width, height, seed=seed + i, max_idle=idle)
        env.reset()
        while not env.done:
            action = bot.act(env)
            _, _, done, info = env.step(action)
            if done:
                lengths.append(env.length)
                if info.get("won"):
                    wins += 1
                else:
                    deaths += 1
                    reasons[info.get("reason", "")] = reasons.get(info.get("reason", ""), 0) + 1
                break
    area = width * height
    return {
        "size": f"{width}x{height}",
        "games": games,
        "mean_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "fill": (sum(lengths) / len(lengths)) / area,
        "win_rate": wins / games,
        "deaths": deaths,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Snake bot on many board sizes.")
    parser.add_argument("--mode", choices=["neural", "perfect", "hybrid", "random"], default="neural")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument(
        "--sizes",
        type=str,
        default="6x6,7x7,8x8,9x10,10x10,12x12,15x16,7x8",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bot = SnakeBot(mode=args.mode, model_path=args.model or None)
    sizes = parse_sizes(args.sizes)
    reports = [run_games(bot, w, h, args.games, args.seed + 17 * (w + h)) for w, h in sizes]
    print(json.dumps({"mode": args.mode, "results": reports}, indent=2))


if __name__ == "__main__":
    main()
