"""Watch a Snake bot play in the terminal."""

from __future__ import annotations

import argparse
import os
import time

from snake.bot import SnakeBot
from snake.env import SnakeEnv


BODY = "██"
FOOD = "● "
EMPTY = "· "
HEAD_CHARS = {0: "▲ ", 1: "▶ ", 2: "▼ ", 3: "◀ "}


def render(env: SnakeEnv) -> str:
    grid = [["\033[90m" + EMPTY + "\033[0m" for _ in range(env.width)] for _ in range(env.height)]
    if env.food is not None:
        fx, fy = env.food
        grid[fy][fx] = "\033[91m" + FOOD + "\033[0m"
    for i, (x, y) in enumerate(env.snake):
        if i == 0:
            grid[y][x] = "\033[92m" + HEAD_CHARS[env.direction] + "\033[0m"
        else:
            grid[y][x] = "\033[32m" + BODY + "\033[0m"
    top = "┌" + "──" * env.width + "┐"
    bottom = "└" + "──" * env.width + "┘"
    rows = ["│" + "".join(row) + "│" for row in grid]
    status = (
        f"len {env.length}/{env.area}  steps {env.steps}  "
        f"board {env.width}x{env.height}"
        + (f"  {env.death_reason}" if env.done and not env.won else "")
        + ("  WIN" if env.won else "")
    )
    return "\n".join([top, *rows, bottom, status])


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Snake with a trained or perfect bot.")
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--mode", choices=["neural", "perfect", "hybrid", "random"], default="perfect")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    idle = 0 if args.mode in {"perfect", "hybrid"} else None
    env = SnakeEnv(args.width, args.height, seed=args.seed, max_idle=idle)
    env.reset()
    bot = SnakeBot(mode=args.mode, model_path=args.model or None)
    steps = 0
    try:
        while True:
            os.system("clear")
            print(render(env))
            print(f"mode={args.mode}")
            if env.done:
                break
            action = bot.act(env)
            env.step(action)
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
