"""Record GIF animations of the never-die bot on every board size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from snake.env import SnakeEnv
from snake.perfect import PerfectBot

SIZES: List[Tuple[int, int]] = [
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 6),
    (8, 8),
    (9, 9),
    (10, 10),
    (12, 8),
    (12, 12),
    (16, 10),
    (20, 12),
]

BG = (7, 16, 24)
GRID = (18, 32, 48)
FOOD = (255, 93, 108)
HEAD = (93, 255, 177)
BODY = (47, 191, 122)


def cell_px(width: int, height: int) -> int:
    longest = max(width, height)
    if longest <= 8:
        return 28
    if longest <= 12:
        return 20
    if longest <= 16:
        return 16
    return 12


def render(env: SnakeEnv, cell: int) -> Image.Image:
    pad = 8
    img = Image.new("RGB", (env.width * cell + pad * 2, env.height * cell + pad * 2), BG)
    draw = ImageDraw.Draw(img)
    for y in range(env.height):
        for x in range(env.width):
            x0 = pad + x * cell
            y0 = pad + y * cell
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=GRID)
    if env.food is not None:
        fx, fy = env.food
        inset = max(2, cell // 4)
        draw.ellipse(
            [
                pad + fx * cell + inset,
                pad + fy * cell + inset,
                pad + fx * cell + cell - inset,
                pad + fy * cell + cell - inset,
            ],
            fill=FOOD,
        )
    n = max(env.length, 1)
    for i, (x, y) in enumerate(env.snake):
        t = i / n
        if i == 0:
            color = HEAD
        else:
            color = (
                int(BODY[0] * (1 - 0.45 * t)),
                int(BODY[1] * (1 - 0.25 * t)),
                int(BODY[2] * (1 - 0.15 * t)),
            )
        inset = max(1, cell // 8) if i else max(1, cell // 10)
        draw.rounded_rectangle(
            [
                pad + x * cell + inset,
                pad + y * cell + inset,
                pad + x * cell + cell - inset - 1,
                pad + y * cell + cell - inset - 1,
            ],
            radius=max(2, cell // 5),
            fill=color,
        )
    return img


def record_game(
    width: int,
    height: int,
    hunt: bool,
    seed: int,
    out: Path,
    max_frames: int = 360,
) -> dict:
    env = SnakeEnv(width, height, seed=seed, max_idle=0)
    env.reset()
    bot = PerfectBot(hunt=hunt)
    cell = cell_px(width, height)
    limit = width * height * 80
    stride = 1
    frames = [render(env, cell)]
    while not env.done and env.steps < limit:
        env.step(bot.act(env))
        if env.done or env.steps % stride == 0:
            frames.append(render(env, cell))
            if len(frames) > max_frames:
                frames = frames[::2]
                stride *= 2
    if not env.won:
        raise RuntimeError(f"{width}x{height} hunt={hunt} failed: {env.death_reason} len={env.length}")
    if len(frames) > max_frames:
        step = int(np.ceil(len(frames) / max_frames))
        kept = frames[::step]
        if kept[-1] is not frames[-1]:
            kept.append(frames[-1])
        frames = kept
    # Freeze the win for a beat.
    frames.extend([frames[-1]] * 8)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
        optimize=True,
    )
    return {
        "size": f"{width}x{height}",
        "hunt": hunt,
        "steps": env.steps,
        "length": env.length,
        "won": env.won,
        "file": str(out),
        "frames": len(frames),
    }


def write_gallery(root: Path, reports: list) -> None:
    by_size = {}
    for row in reports:
        by_size.setdefault(row["size"], {})[row["hunt"]] = row

    def card(row, label: str) -> str:
        rel = Path(row["file"]).relative_to(root)
        return (
            f'<figure><img src="{rel.as_posix()}" alt="{row["size"]} {label}" />'
            f"<figcaption>{label} · {row['steps']} steps</figcaption></figure>"
        )

    blocks = []
    for size, variants in by_size.items():
        fast = variants.get(True)
        rail = variants.get(False)
        saved = ""
        if fast and rail:
            pct = 100 * (rail["steps"] - fast["steps"]) / rail["steps"]
            saved = f' · fast saves {rail["steps"] - fast["steps"]} steps ({pct:.0f}%)'
        figs = []
        if rail:
            figs.append(card(rail, "rail"))
        if fast:
            figs.append(card(fast, "fast"))
        blocks.append(
            f"<article><h2>{size}{saved}</h2><div class=\"pair\">{''.join(figs)}</div></article>"
        )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<title>Snake animations</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #0b1220; color: #e7eefc; margin: 24px; }}
h1 {{ font-size: 1.2rem; }}
h2 {{ font-size: 1rem; margin: 0 0 8px; }}
p {{ color: #93a0bb; max-width: 720px; }}
article {{ margin: 24px 0; }}
.pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
figure {{ margin: 0; background: #121b2d; border-radius: 12px; padding: 10px; }}
img {{ width: 100%; image-rendering: pixelated; }}
figcaption {{ font-size: 12px; color: #93a0bb; margin-top: 8px; }}
</style></head>
<body>
<h1>Never-die Snake, every board size</h1>
<p>Rail follows one covering cycle, so every lap looks the same. Fast still never dies: it only jumps ahead on that cycle when the path from the new head to the tail is empty, which cuts about 25–40% of the steps.</p>
{''.join(blocks)}
</body></html>
"""
    (root / "index.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record GIFs of the never-die bot.")
    parser.add_argument("--out", default="animations")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--style", choices=["fast", "rail", "both"], default="both")
    args = parser.parse_args()
    root = Path(args.out)
    reports = []
    styles = []
    if args.style in {"fast", "both"}:
        styles.append(("fast", True))
    if args.style in {"rail", "both"}:
        styles.append(("rail", False))
    for hunt_name, hunt in styles:
        for w, h in SIZES:
            path = root / hunt_name / f"snake_{w}x{h}.gif"
            print(f"recording {hunt_name} {w}x{h} ...", flush=True)
            info = record_game(w, h, hunt, args.seed, path)
            print(f"  steps={info['steps']} -> {path}", flush=True)
            reports.append(info)
    (root / "manifest.json").write_text(json.dumps(reports, indent=2))
    write_gallery(root, reports)
    print(f"Wrote {root / 'manifest.json'} and {root / 'index.html'}", flush=True)
    by_size = {}
    for row in reports:
        by_size.setdefault(row["size"], {})[row["hunt"]] = row["steps"]
    print("\nsteps to fill (lower is better)", flush=True)
    for size, vals in by_size.items():
        rail = vals.get(False)
        fast = vals.get(True)
        if rail is not None and fast is not None:
            print(f"  {size:6s}  rail {rail:5d}  fast {fast:5d}  saved {rail - fast}", flush=True)
        else:
            print(f"  {size:6s}  {vals}", flush=True)


if __name__ == "__main__":
    main()
