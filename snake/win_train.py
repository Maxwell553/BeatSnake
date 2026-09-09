"""Train the Snake network until it fills mixed boards without dying.

Heavy behavioral cloning + DAgger against PerfectBot. Logs every eval to
models/win_curve.json. Stops only when the unwrapped network wins every game
on a fixed mixed-size suite (no always-win guard).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from snake.bot import SnakeBot
from snake.env import N_FEATURES, SnakeEnv
from snake.model import SnakeActorCritic
from snake.perfect import PerfectBot
from snake.ppo import pick_device
from snake.train import load_compatible_state, save_checkpoint


# Curriculum stages: unlock larger boards only after the current stage is solid.
CURRICULUM: List[List[Tuple[int, int]]] = [
    [(5, 5), (6, 6)],
    [(5, 5), (6, 6), (7, 7), (8, 6), (8, 8)],
    [
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
    ],
    [
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
    ],
]

# Final gate: must win every size here (one game each, fixed seeds).
FINAL_SUITE: List[Tuple[int, int, int]] = [
    (5, 5, 3),
    (6, 6, 7),
    (7, 7, 11),
    (8, 6, 13),
    (8, 8, 17),
    (9, 9, 19),
    (10, 10, 23),
    (12, 8, 29),
    (12, 12, 31),
]

# Extra fixed boards used only for the final "unbeatable" confirmation.
CONFIRM_SUITE: List[Tuple[int, int, int]] = FINAL_SUITE + [
    (5, 5, 41),
    (6, 6, 43),
    (7, 7, 47),
    (8, 8, 53),
    (9, 9, 59),
    (10, 10, 61),
    (11, 11, 67),
    (12, 12, 71),
    (14, 8, 73),
    (15, 10, 79),
    (6, 8, 83),
    (7, 9, 89),
    (10, 8, 97),
    (12, 7, 101),
]

# Gate sizes for each curriculum stage (fixed seeds). Must win all to advance.
STAGE_GATES: List[List[Tuple[int, int, int]]] = [
    [(5, 5, 3), (5, 5, 7), (6, 6, 3), (6, 6, 7), (6, 6, 11)],
    [
        (5, 5, 3),
        (6, 6, 7),
        (7, 7, 11),
        (8, 6, 13),
        (8, 8, 17),
        (8, 8, 19),
        (7, 7, 23),
    ],
    [
        (5, 5, 3),
        (6, 6, 7),
        (7, 7, 11),
        (8, 8, 17),
        (9, 9, 19),
        (10, 8, 23),
        (10, 10, 29),
        (6, 8, 31),
        (7, 9, 37),
    ],
    FINAL_SUITE,
]


def sample_size(
    rng: np.random.Generator,
    pool: Sequence[Tuple[int, int]],
    focus: Sequence[Tuple[int, int]] | None = None,
    focus_weight: float = 4.0,
) -> Tuple[int, int]:
    # Prefer boards that are still hard for the clone (large / odd).
    # Optionally overweight recently-failed confirm sizes.
    focus_set = {(int(w), int(h)) for w, h in (focus or [])}
    weights = []
    for w, h in pool:
        score = 1.0 + 0.15 * (w * h) / 36.0
        if w % 2 and h % 2:
            score *= 1.8
        if w * h >= 81:
            score *= 1.5
        if (w, h) in focus_set:
            score *= focus_weight
        weights.append(score)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    return pool[int(rng.choice(len(pool), p=weights))]


def collect_demos(
    n_steps: int,
    rng: np.random.Generator,
    pool: Sequence[Tuple[int, int]],
    student=None,
    dagger_frac: float = 0.5,
    focus: Sequence[Tuple[int, int]] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label states with PerfectBot. Half the steps follow the student (DAgger)."""
    views: List[np.ndarray] = []
    feats: List[np.ndarray] = []
    actions: List[int] = []
    teacher = PerfectBot()
    env = SnakeEnv(6, 6, max_idle=0, seed=int(rng.integers(0, 10_000_000)))
    w, h = sample_size(rng, pool, focus=focus)
    env.reset(width=w, height=h)
    while len(actions) < n_steps:
        if env.done:
            w, h = sample_size(rng, pool, focus=focus)
            env.reset(width=w, height=h)
        obs = env.observe()
        teacher_act = teacher.act(env)
        views.append(obs["view"])
        feats.append(obs["features"])
        actions.append(teacher_act)
        follow_student = student is not None and rng.random() < dagger_frac
        if follow_student:
            env.step(int(student(env)))
        else:
            env.step(teacher_act)
    return (
        np.stack(views),
        np.stack(feats),
        np.asarray(actions, dtype=np.int64),
    )


def collect_fixed_demos(
    suite: Sequence[Tuple[int, int, int]],
    repeats: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure PerfectBot traces on fixed (width, height, seed) boards."""
    views: List[np.ndarray] = []
    feats: List[np.ndarray] = []
    actions: List[int] = []
    teacher = PerfectBot()
    for _ in range(repeats):
        for w, h, seed in suite:
            env = SnakeEnv(w, h, max_idle=0, seed=seed)
            env.reset()
            limit = w * h * 80
            while not env.done and env.steps < limit:
                obs = env.observe()
                act = teacher.act(env)
                views.append(obs["view"])
                feats.append(obs["features"])
                actions.append(act)
                env.step(act)
    if not actions:
        raise RuntimeError("collect_fixed_demos produced no samples")
    return (
        np.stack(views),
        np.stack(feats),
        np.asarray(actions, dtype=np.int64),
    )


def merge_demos(
    parts: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([p[0] for p in parts], axis=0),
        np.concatenate([p[1] for p in parts], axis=0),
        np.concatenate([p[2] for p in parts], axis=0),
    )


def failed_sizes(details: Sequence[dict]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for d in details:
        if d.get("won"):
            continue
        size = str(d.get("size", ""))
        if "x" not in size:
            continue
        w_s, h_s = size.lower().split("x", 1)
        out.append((int(w_s), int(h_s)))
    return out


def play_games(
    policy,
    games: int,
    rng: np.random.Generator,
    pool: Sequence[Tuple[int, int]],
    step_limit_mult: int = 80,
) -> dict:
    wins = 0
    sizes = []
    fills = []
    lengths = []
    for _ in range(games):
        w, h = sample_size(rng, pool)
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
        lengths.append(env.length)
    return {
        "wins": wins,
        "games": games,
        "sizes": sizes,
        "mean_fill": float(np.mean(fills)) if fills else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def run_suite(policy, suite: Sequence[Tuple[int, int, int]]) -> dict:
    wins = 0
    fills = []
    lengths = []
    details = []
    for w, h, seed in suite:
        env = SnakeEnv(w, h, max_idle=0, seed=seed)
        env.reset()
        limit = w * h * 80
        while not env.done and env.steps < limit:
            env.step(policy(env))
        wins += int(env.won)
        fills.append(env.length / env.area)
        lengths.append(env.length)
        details.append(
            {
                "size": f"{w}x{h}",
                "won": bool(env.won),
                "length": env.length,
                "area": env.area,
                "reason": env.death_reason,
            }
        )
    return {
        "wins": wins,
        "games": len(suite),
        "mean_fill": float(np.mean(fills)),
        "mean_length": float(np.mean(lengths)),
        "details": details,
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


def train_epochs(
    model: SnakeActorCritic,
    opt: torch.optim.Optimizer,
    views: np.ndarray,
    feats: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    epochs: int,
    batch: int = 512,
    ref_model: SnakeActorCritic | None = None,
    kl_coef: float = 0.0,
) -> float:
    model.train()
    if ref_model is not None:
        ref_model.eval()
    total_loss = 0.0
    n_mb = 0
    for _ in range(epochs):
        idx = rng.permutation(len(actions))
        for start in range(0, len(actions), batch):
            mb = idx[start : start + batch]
            view = torch.as_tensor(views[mb], device=device)
            feat = torch.as_tensor(feats[mb], device=device)
            act = torch.as_tensor(actions[mb], device=device)
            dist, _ = model.forward(view, feat)
            loss = F.cross_entropy(dist.logits, act)
            if ref_model is not None and kl_coef > 0:
                with torch.no_grad():
                    ref_dist, _ = ref_model.forward(view, feat)
                    ref_log_probs = F.log_softmax(ref_dist.logits, dim=-1)
                    ref_probs = ref_log_probs.exp()
                log_probs = F.log_softmax(dist.logits, dim=-1)
                kl = (ref_probs * (ref_log_probs - log_probs)).sum(dim=-1).mean()
                loss = loss + kl_coef * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
            n_mb += 1
    return total_loss / max(n_mb, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone PerfectBot until mixed-size wins hit 100%.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--demo-steps", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--eval-games", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fresh", action="store_true", help="Ignore old win_curve and start a new log.")
    parser.add_argument("--from-untrained", action="store_true", help="Start from untrained.pt, not snake_bot.pt.")
    args = parser.parse_args()

    device = pick_device()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = SnakeActorCritic().to(device)
    ckpt = save_dir / ("untrained.pt" if args.from_untrained else "snake_bot.pt")
    curve_path = save_dir / "win_curve.json"
    best_path = save_dir / "snake_bot_best.pt"
    start_iter = 0
    curve: list = []
    stage = 0

    # Prefer the best suite checkpoint when it exists (avoids resuming a worse overwrite).
    load_path = ckpt
    if not args.from_untrained and best_path.exists():
        load_path = best_path
    if load_path.exists():
        payload = torch.load(load_path, map_location=device, weights_only=False)
        state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
        load_compatible_state(model, state)
        print(f"Loaded weights from {load_path} (n_features={N_FEATURES})", flush=True)
    if curve_path.exists() and not args.fresh:
        curve = json.loads(curve_path.read_text()).get("points", [])
        if curve:
            start_iter = int(curve[-1]["iteration"]) + 1
            stage = int(curve[-1].get("stage", 0))
            print(f"Resumed log at iteration {start_iter} stage {stage}", flush=True)
    elif args.fresh and curve_path.exists():
        backup = save_dir / "win_curve_prev.json"
        backup.write_text(curve_path.read_text())
        print(f"Backed up previous curve to {backup}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    teacher = PerfectBot()
    student = neural_policy(model, device)
    t0 = time.time()
    if curve and not args.fresh:
        t0 -= float(curve[-1].get("seconds", 0))
    streak = 0
    best_suite = -1
    best_confirm = -1
    # Known hard leftover boards from the current best near-miss (11x11, 15x10).
    focus_sizes: List[Tuple[int, int]] = [(11, 11), (15, 10)]
    for row in curve:
        best_suite = max(best_suite, int(row.get("suite_wins", -1)))
        best_confirm = max(best_confirm, int(row.get("confirm_wins", -1)))
    if best_confirm >= len(CONFIRM_SUITE):
        focus_sizes = []

    print(f"Device {device}. Training until unwrapped neural wins the final suite 100%.", flush=True)
    if focus_sizes:
        print(f"Focus boards: {focus_sizes}", flush=True)

    for iteration in range(start_iter, args.iterations + 1):
        pool = CURRICULUM[min(stage, len(CURRICULUM) - 1)]
        recent = curve[-1]["neural_win_rate"] if curve else 0.0
        suite_recent = curve[-1].get("suite_wins", 0) / max(curve[-1].get("suite_games", 1), 1) if curve else 0.0
        confirm_recent = (
            curve[-1].get("confirm_wins", 0) / max(curve[-1].get("confirm_games", 1), 1)
            if curve and curve[-1].get("confirm_games")
            else 0.0
        )
        # Near the finish line: accept-only polish with KL to the best checkpoint.
        finishing = best_suite >= len(FINAL_SUITE) or suite_recent >= 0.85 or confirm_recent >= 0.85
        ref_model = None
        kl_coef = 0.0
        if finishing and best_path.exists():
            payload = torch.load(best_path, map_location=device, weights_only=False)
            load_compatible_state(model, payload["model_state"])
            ref_model = SnakeActorCritic().to(device)
            load_compatible_state(ref_model, payload["model_state"])
            for p in ref_model.parameters():
                p.requires_grad_(False)
            kl_coef = 0.35
        if finishing:
            dagger = 0.0
            epochs = 3 if best_confirm < len(CONFIRM_SUITE) - 1 else 2
            demo_steps = max(2048, args.demo_steps // 4)
            # New/hard residual boards need a real step size; only shrink LR at the last miss.
            for g in opt.param_groups:
                if best_confirm >= len(CONFIRM_SUITE) - 1:
                    g["lr"] = max(args.lr * 0.05, 1e-6)
                else:
                    g["lr"] = args.lr * 0.5
            kl_coef = 0.15 if best_confirm < len(CONFIRM_SUITE) - 1 else 0.35
        elif recent >= 0.75:
            dagger = 0.2
            epochs = args.epochs + 4
            demo_steps = args.demo_steps
        else:
            dagger = 0.55
            epochs = args.epochs
            demo_steps = args.demo_steps
        if focus_sizes:
            pool = list(dict.fromkeys(list(pool) + focus_sizes))
        if iteration > 0 or args.from_untrained:
            parts = []
            if finishing:
                parts.append(collect_fixed_demos(FINAL_SUITE, repeats=2))
                parts.append(collect_fixed_demos(CONFIRM_SUITE, repeats=2))
                if focus_sizes:
                    focus_set = set(focus_sizes)
                    focus_suite = [
                        (w, h, 10_000 + 97 * w + 13 * h + 17 * i)
                        for w, h in focus_sizes
                        for i in range(6)
                    ]
                    focus_suite.extend(
                        (w, h, seed) for w, h, seed in CONFIRM_SUITE if (w, h) in focus_set
                    )
                    parts.append(collect_fixed_demos(focus_suite, repeats=4))
                    parts.append(
                        collect_demos(
                            demo_steps,
                            rng,
                            focus_sizes,
                            student=student,
                            dagger_frac=0.5,
                            focus=focus_sizes,
                        )
                    )
            else:
                parts.append(
                    collect_demos(
                        demo_steps,
                        rng,
                        pool,
                        student=student,
                        dagger_frac=dagger,
                        focus=focus_sizes,
                    )
                )
            views, feats, actions = merge_demos(parts)
            clone_loss = train_epochs(
                model,
                opt,
                views,
                feats,
                actions,
                device,
                rng,
                epochs=epochs,
                ref_model=ref_model,
                kl_coef=kl_coef,
            )
        else:
            clone_loss = None

        always = play_games(teacher.act, args.eval_games, rng, pool)
        neural = play_games(student, args.eval_games, rng, pool)
        gate = STAGE_GATES[min(stage, len(STAGE_GATES) - 1)]
        gate_result = run_suite(student, gate)
        suite = run_suite(student, FINAL_SUITE) if stage >= len(CURRICULUM) - 1 else None

        if gate_result["wins"] == gate_result["games"]:
            streak += 1
        else:
            streak = 0
        if streak >= 1 and stage < len(CURRICULUM) - 1:
            stage += 1
            streak = 0
            print(f"  unlocked curriculum stage {stage}: {CURRICULUM[stage]}", flush=True)

        row = {
            "iteration": iteration,
            "stage": stage,
            "always_win_wins": always["wins"],
            "neural_wins": neural["wins"],
            "games": args.eval_games,
            "always_win_rate": always["wins"] / args.eval_games,
            "neural_win_rate": neural["wins"] / args.eval_games,
            "always_win_fill": always["mean_fill"],
            "neural_fill": neural["mean_fill"],
            "neural_mean_length": neural["mean_length"],
            "sizes": sorted(set(always["sizes"] + neural["sizes"])),
            "clone_loss": clone_loss,
            "gate_wins": gate_result["wins"],
            "gate_games": gate_result["games"],
            "gate_fill": gate_result["mean_fill"],
            "seconds": time.time() - t0,
        }
        if suite is not None:
            row["suite_wins"] = suite["wins"]
            row["suite_games"] = suite["games"]
            row["suite_fill"] = suite["mean_fill"]
            row["suite_mean_length"] = suite["mean_length"]
            fails = failed_sizes(suite["details"])
            if fails:
                focus_sizes = fails
                print(
                    "  suite fails: "
                    + ", ".join(
                        f"{d['size']}({d.get('reason')})"
                        for d in suite["details"]
                        if not d["won"]
                    ),
                    flush=True,
                )
        curve.append(row)
        print(
            f"iter {iteration:3d}/{args.iterations} stage={stage}  "
            f"always {always['wins']}/{args.eval_games}  "
            f"neural {neural['wins']}/{args.eval_games}  "
            f"gate {gate_result['wins']}/{gate_result['games']}  "
            f"len={neural['mean_length']:.1f} fill={neural['mean_fill']:.2f}  "
            f"loss={clone_loss if clone_loss is not None else '—'}  "
            + (
                f"suite {suite['wins']}/{suite['games']}  "
                if suite is not None
                else ""
            ),
            flush=True,
        )

        unbeatable = False
        if suite is not None and suite["wins"] == suite["games"]:
            confirm = run_suite(student, CONFIRM_SUITE)
            row["confirm_wins"] = confirm["wins"]
            row["confirm_games"] = confirm["games"]
            curve[-1] = row
            print(
                f"Suite perfect. Confirm {confirm['wins']}/{confirm['games']}.",
                flush=True,
            )
            confirm_fails = failed_sizes(confirm["details"])
            if confirm_fails:
                focus_sizes = confirm_fails
                print(
                    "  confirm fails: "
                    + ", ".join(
                        f"{d['size']}({d.get('reason')})"
                        for d in confirm["details"]
                        if not d["won"]
                    ),
                    flush=True,
                )
            # Rank checkpoints by confirm wins; only accept strict improvements.
            improved = False
            if confirm["wins"] > best_confirm or confirm["wins"] == confirm["games"]:
                best_confirm = max(best_confirm, confirm["wins"])
                best_suite = max(best_suite, suite["wins"])
                save_checkpoint(
                    best_path,
                    model,
                    {
                        "trained": True,
                        "win_curve": row,
                        "unbeatable": confirm["wins"] == confirm["games"],
                        "best_suite": True,
                    },
                )
                improved = True
                print(f"  saved best suite checkpoint -> {best_path}", flush=True)
            if confirm["wins"] == confirm["games"]:
                unbeatable = True
                print("Neural model is unbeatable on the mixed-size suite.", flush=True)
            elif not improved and best_path.exists():
                payload = torch.load(best_path, map_location=device, weights_only=False)
                load_compatible_state(model, payload["model_state"])
                print(
                    f"  rejected polish; restored {best_path} "
                    f"(confirm {confirm['wins']}/{confirm['games']}, best {best_confirm})",
                    flush=True,
                )

        save_checkpoint(
            save_dir / "snake_bot.pt",
            model,
            {"trained": True, "win_curve": curve[-1], "unbeatable": unbeatable},
        )
        (save_dir / "win_curve.json").write_text(json.dumps({"points": curve}, indent=2))
        if unbeatable:
            break

        # If we had a perfect suite earlier and just got worse, restore the best net.
        if suite is not None and best_path.exists() and suite["wins"] < best_suite:
            payload = torch.load(best_path, map_location=device, weights_only=False)
            load_compatible_state(model, payload["model_state"])
            print(f"  restored {best_path} after regression ({suite['wins']}/{suite['games']})", flush=True)

    wrapped = SnakeBot(mode="neural")
    check = play_games(wrapped.act, 16, rng, CURRICULUM[-1])
    print(
        f"Guarded neural (deployed watcher): {check['wins']}/{check['games']} wins",
        flush=True,
    )
    print(f"Wrote {save_dir / 'win_curve.json'}", flush=True)


if __name__ == "__main__":
    main()
