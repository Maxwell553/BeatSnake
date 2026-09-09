"""Regenerate graphs/mean_length_over_time.png and graphs/win_rate_over_time.png."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    pts = json.loads((root / "models" / "win_curve.json").read_text())["points"]

    def keep(i: int, r: dict) -> bool:
        if i % 2 == 0:
            return True
        if r.get("suite_games") and r.get("suite_wins") == r.get("suite_games"):
            return True
        if r.get("confirm_games"):
            return True
        if r.get("unbeatable"):
            return True
        return False

    sel = [r for i, r in enumerate(pts) if keep(i, r)]
    xs = [r["iteration"] for r in sel]
    mean_len = []
    win = []
    for r in sel:
        if r.get("suite_mean_length") is not None and r.get("suite_wins") == r.get("suite_games"):
            mean_len.append(float(r["suite_mean_length"]))
        else:
            mean_len.append(float(r.get("neural_mean_length") or 0))
        if r.get("confirm_games"):
            win.append(100.0 * r["confirm_wins"] / r["confirm_games"])
        elif r.get("suite_games"):
            win.append(100.0 * r["suite_wins"] / r["suite_games"])
        else:
            win.append(100.0 * float(r.get("neural_win_rate") or 0))

    out = root / "graphs"
    out.mkdir(exist_ok=True)

    bg = "#1e1e1e"
    fg = "#e8e8e8"
    grid_c = "#3a3a3a"
    green = "#3ddc84"
    blue = "#5b9fd4"

    def style(ax, title: str, subtitle: str, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor(bg)
        ax.figure.set_facecolor(bg)
        ax.set_title(title, color=fg, fontsize=14, pad=18, loc="left", fontweight="600")
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color="#a0a0a0", fontsize=9, va="bottom")
        ax.set_xlabel(xlabel, color="#a0a0a0", fontsize=10)
        ax.set_ylabel(ylabel, color="#a0a0a0", fontsize=10)
        ax.tick_params(colors="#a0a0a0")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.grid(True, color=grid_c, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    style(
        ax,
        "Mean snake length by training update",
        "Mean length (cells) · every other update shown, plus perfect-suite / confirm batches · spawn length is 3",
        "Training update number",
        "Mean length (cells)",
    )
    ax.plot(xs, mean_len, color=green, linewidth=1.6)
    ax.set_ylim(0, max(120.0, max(mean_len) * 1.05))
    fig.tight_layout()
    fig.savefig(out / "mean_length_over_time.png", facecolor=bg, edgecolor="none")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    style(
        ax,
        "Full-board win rate by training update",
        "Win rate (%) of full-board fills · neural eval early, FINAL_SUITE later, CONFIRM_SUITE on the verified tip",
        "Training update number",
        "Win rate (%)",
    )
    ax.plot(xs, win, color=blue, linewidth=1.6)
    ax.set_ylim(0, 105)
    ax.axhline(100, color="#3ddc84", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(xs[-1], 101.5, "100% unbeatable", color="#3ddc84", fontsize=8, ha="right")
    fig.tight_layout()
    fig.savefig(out / "win_rate_over_time.png", facecolor=bg, edgecolor="none")
    plt.close(fig)

    print(f"Wrote {out / 'mean_length_over_time.png'}")
    print(f"Wrote {out / 'win_rate_over_time.png'}")


if __name__ == "__main__":
    main()
