"""Record GIF animations of the never-die bot on every board size.

Frames are drawn in the Google Snake look: light-green checkerboard, thick
dark-green border, solid blue snake with eyes, and a red apple with a leaf.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from snake.env import DIRS, SnakeEnv
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

# Google Snake palette.
BOARD_A = (170, 215, 81)  # #aad751
BOARD_B = (162, 209, 73)  # #a2d149
BORDER = (87, 138, 52)  # #578a34
SNAKE = (78, 124, 246)  # #4e7cf6
EYE_WHITE = (255, 255, 255)
APPLE = (231, 71, 29)  # #e7471d
APPLE_SHADOW = (154, 192, 66)
STEM = (92, 64, 51)
LEAF = (94, 178, 58)
OUTSIDE = (74, 117, 44)  # #4a752c


def cell_px(width: int, height: int) -> int:
    longest = max(width, height)
    if longest <= 8:
        return 36
    if longest <= 12:
        return 28
    if longest <= 16:
        return 22
    return 16


def _cell_box(pad: int, cell: int, x: int, y: int, inset: float = 0.0) -> List[float]:
    m = cell * inset
    return [
        pad + x * cell + m,
        pad + y * cell + m,
        pad + x * cell + cell - m - 1,
        pad + y * cell + cell - m - 1,
    ]


def _draw_apple(draw: ImageDraw.ImageDraw, pad: int, cell: int, fx: int, fy: int) -> None:
    # Soft checkerboard-tinted shadow under the fruit.
    shadow = _cell_box(pad, cell, fx, fy, 0.22)
    shadow[1] += cell * 0.08
    shadow[3] += cell * 0.08
    draw.ellipse(shadow, fill=APPLE_SHADOW)

    body = _cell_box(pad, cell, fx, fy, 0.18)
    draw.ellipse(body, fill=APPLE)

    cx = pad + fx * cell + cell / 2
    top = pad + fy * cell + cell * 0.18
    stem_w = max(1.5, cell * 0.06)
    draw.line([(cx, top), (cx, top - cell * 0.12)], fill=STEM, width=max(1, int(stem_w)))

    leaf_r = max(2.0, cell * 0.16)
    draw.ellipse(
        [cx + cell * 0.02, top - cell * 0.22, cx + cell * 0.02 + leaf_r * 2, top - cell * 0.02],
        fill=LEAF,
    )


def _draw_snake(draw: ImageDraw.ImageDraw, env: SnakeEnv, pad: int, cell: int) -> None:
    if not env.snake:
        return
    inset = 0.12
    radius = max(2, int(cell * 0.42))

    # Draw from tail to head so the head paints on top. Adjacent cells share a
    # rounded connector so the body reads as one continuous Google-style snake.
    for i in range(len(env.snake) - 1, -1, -1):
        x, y = env.snake[i]
        box = _cell_box(pad, cell, x, y, inset)
        draw.rounded_rectangle(box, radius=radius, fill=SNAKE)
        if i + 1 < len(env.snake):
            nx, ny = env.snake[i + 1]
            if abs(nx - x) + abs(ny - y) == 1:
                x0 = pad + min(x, nx) * cell + cell * inset
                y0 = pad + min(y, ny) * cell + cell * inset
                x1 = pad + max(x, nx) * cell + cell * (1 - inset) - 1
                y1 = pad + max(y, ny) * cell + cell * (1 - inset) - 1
                draw.rectangle([x0, y0, x1, y1], fill=SNAKE)

    # Eyes on the head, looking in the facing direction.
    hx, hy = env.head
    cx = pad + hx * cell + cell / 2
    cy = pad + hy * cell + cell / 2
    dx, dy = DIRS[env.direction]
    eye_r = max(2.0, cell * 0.16)
    pupil_r = max(1.0, cell * 0.07)
    along = cell * 0.14
    across = cell * 0.18

    # Perpendicular offset for left/right eyes.
    px, py = -dy, dx
    centers = [
        (cx + dx * along + px * across, cy + dy * along + py * across),
        (cx + dx * along - px * across, cy + dy * along - py * across),
    ]
    for ex, ey in centers:
        draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r], fill=EYE_WHITE)
        px_ = ex + dx * eye_r * 0.35
        py_ = ey + dy * eye_r * 0.35
        draw.ellipse(
            [px_ - pupil_r, py_ - pupil_r, px_ + pupil_r, py_ + pupil_r],
            fill=SNAKE,
        )


def render(env: SnakeEnv, cell: int) -> Image.Image:
    border = max(8, cell // 2)
    pad = border
    w = env.width * cell + pad * 2
    h = env.height * cell + pad * 2
    img = Image.new("RGB", (w, h), OUTSIDE)
    draw = ImageDraw.Draw(img)

    # Thick border, then the two-tone checkerboard playfield.
    draw.rectangle([0, 0, w - 1, h - 1], fill=BORDER)
    for y in range(env.height):
        for x in range(env.width):
            color = BOARD_A if (x + y) % 2 == 0 else BOARD_B
            x0 = pad + x * cell
            y0 = pad + y * cell
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=color)

    if env.food is not None:
        _draw_apple(draw, pad, cell, env.food[0], env.food[1])
    _draw_snake(draw, env, pad, cell)
    return img


def record_game(
    width: int,
    height: int,
    hunt: bool,
    seed: int,
    out: Path,
    max_frames: int = 360,
    policy=None,
    require_win: bool = True,
    label: str = "perfect",
    max_idle: int = 0,
    step_limit_mult: int = 80,
) -> dict:
    env = SnakeEnv(width, height, seed=seed, max_idle=max_idle)
    env.reset()
    bot = policy if policy is not None else PerfectBot(hunt=hunt)
    cell = cell_px(width, height)
    limit = width * height * step_limit_mult
    stride = 1
    frames = [render(env, cell)]
    while not env.done and env.steps < limit:
        env.step(bot.act(env))
        if env.done or env.steps % stride == 0:
            frames.append(render(env, cell))
            if len(frames) > max_frames:
                frames = frames[::2]
                stride *= 2
    if require_win and not env.won:
        raise RuntimeError(f"{width}x{height} {label} failed: {env.death_reason} len={env.length}")
    if len(frames) > max_frames:
        step = int(np.ceil(len(frames) / max_frames))
        kept = frames[::step]
        if kept[-1] is not frames[-1]:
            kept.append(frames[-1])
        frames = kept
    # Freeze the last frame for a beat.
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
        "label": label,
        "steps": env.steps,
        "length": env.length,
        "won": env.won,
        "reason": env.death_reason or ("timeout" if not env.done else ""),
        "file": str(out),
        "frames": len(frames),
    }


def write_gallery(root: Path, reports: list) -> None:
    by_size = {}
    for row in reports:
        key = row.get("label") or ("fast" if row.get("hunt") else "rail")
        by_size.setdefault(row["size"], {})[key] = row

    def card(row, label: str) -> str:
        rel = Path(row["file"]).relative_to(root)
        extra = f" · won" if row.get("won") else f" · {row.get('reason') or 'ended'} len={row.get('length')}"
        return (
            f'<figure><img src="{rel.as_posix()}" alt="{row["size"]} {label}" />'
            f"<figcaption>{label} · {row['steps']} steps{extra}</figcaption></figure>"
        )

    blocks = []
    for size, variants in by_size.items():
        order = [k for k in ("rail", "fast", "untrained") if k in variants]
        figs = [card(variants[k], k) for k in order]
        saved = ""
        if "fast" in variants and "rail" in variants:
            rail = variants["rail"]["steps"]
            fast = variants["fast"]["steps"]
            pct = 100 * (rail - fast) / rail
            saved = f" · fast saves {rail - fast} steps ({pct:.0f}%)"
        blocks.append(
            f"<article><h2>{size}{saved}</h2><div class=\"pair\">{''.join(figs)}</div></article>"
        )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<title>Snake animations</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #4a752c; color: #f4f8e8; margin: 24px; }}
h1 {{ font-size: 1.2rem; }}
h2 {{ font-size: 1rem; margin: 0 0 8px; color: #e8f0d0; }}
p {{ color: #d5e4b0; max-width: 720px; }}
article {{ margin: 24px 0; }}
.pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
figure {{ margin: 0; background: #578a34; border-radius: 12px; padding: 10px; }}
img {{ width: 100%; image-rendering: auto; border-radius: 6px; }}
figcaption {{ font-size: 12px; color: #d5e4b0; margin-top: 8px; }}
</style></head>
<body>
<h1>Snake animations, every board size</h1>
<p>Drawn like Google Snake. Rail / fast are the never-die covering cycle. Untrained is the blank network (dies quickly).</p>
{''.join(blocks)}
</body></html>
"""
    (root / "index.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record GIFs of Snake bots.")
    parser.add_argument("--out", default="animations")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--style",
        choices=["fast", "rail", "both", "untrained"],
        default="both",
    )
    parser.add_argument("--model", default="", help="Weights for untrained style (default models/untrained.pt)")
    args = parser.parse_args()
    root = Path(args.out)
    reports = []

    if args.style == "untrained":
        from snake.bot import SnakeBot

        # Untrained net has no always-win guard: use raw network actions.
        bot = SnakeBot(mode="neural", model_path=args.model or "models/untrained.pt")

        class RawNet:
            def act(self, env):
                return bot._neural_action(env)

        policy = RawNet()
        for w, h in SIZES:
            path = root / "untrained" / f"snake_{w}x{h}.gif"
            print(f"recording untrained {w}x{h} ...", flush=True)
            info = record_game(
                w,
                h,
                True,
                args.seed,
                path,
                policy=policy,
                require_win=False,
                label="untrained",
                max_idle=w * h * 2,
                step_limit_mult=8,
            )
            print(
                f"  steps={info['steps']} won={info['won']} len={info['length']} -> {path}",
                flush=True,
            )
            reports.append(info)
    else:
        styles = []
        if args.style in {"fast", "both"}:
            styles.append(("fast", True))
        if args.style in {"rail", "both"}:
            styles.append(("rail", False))
        for hunt_name, hunt in styles:
            for w, h in SIZES:
                path = root / hunt_name / f"snake_{w}x{h}.gif"
                print(f"recording {hunt_name} {w}x{h} ...", flush=True)
                info = record_game(w, h, hunt, args.seed, path, label=hunt_name)
                print(f"  steps={info['steps']} -> {path}", flush=True)
                reports.append(info)

    # Merge with existing manifest entries for other styles when recording one style.
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and args.style in {"untrained", "fast", "rail"}:
        prev = json.loads(manifest_path.read_text())
        keep_labels = {
            "untrained": {"fast", "rail"},
            "fast": {"rail", "untrained"},
            "rail": {"fast", "untrained"},
        }[args.style]
        prev = [r for r in prev if (r.get("label") or ("fast" if r.get("hunt") else "rail")) in keep_labels]
        reports = prev + reports

    manifest_path.write_text(json.dumps(reports, indent=2))
    write_gallery(root, reports)
    print(f"Wrote {manifest_path} and {root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
