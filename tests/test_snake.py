"""Sanity checks for the environment, Hamiltonian cycle, and untrained network."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from snake.env import CHANNELS, N_FEATURES, VIEW_SIZE, SnakeEnv
from snake.model import SnakeActorCritic
from snake.perfect import PerfectBot, hamiltonian_cycle


class EnvTests(unittest.TestCase):
    def test_eat_grows_and_death_on_wall(self) -> None:
        env = SnakeEnv(6, 6, seed=1)
        obs = env.reset()
        self.assertEqual(obs["view"].shape, (CHANNELS, VIEW_SIZE, VIEW_SIZE))
        self.assertEqual(obs["features"].shape, (N_FEATURES,))
        start = env.length
        # Force food just in front of the head and go straight.
        hx, hy = env.head
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][env.direction]
        env.food = (hx + dx, hy + dy)
        obs, reward, done, info = env.step(1)
        self.assertTrue(info["ate"])
        self.assertEqual(reward, 1.0)
        self.assertEqual(env.length, start + 1)
        self.assertFalse(done)

    def test_wall_death_is_punished(self) -> None:
        env = SnakeEnv(6, 6, seed=0)
        env.reset()
        env.snake = [(0, 0), (1, 0), (2, 0)]
        env.direction = 3  # facing left, next straight hits the wall
        _, reward, done, info = env.step(1)
        self.assertTrue(done)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info["reason"], "wall")

    def test_relative_controls(self) -> None:
        env = SnakeEnv(8, 8, seed=2)
        env.reset()
        env.snake = [(4, 4), (3, 4), (2, 4)]
        env.direction = 1
        env.food = (0, 0)
        env.step(2)  # turn right -> face down
        self.assertEqual(env.direction, 2)
        self.assertEqual(env.head, (4, 5))


class CycleTests(unittest.TestCase):
    def test_even_dimension_is_a_cycle(self) -> None:
        for w, h in [(4, 4), (6, 5), (5, 8), (10, 10)]:
            cycle = hamiltonian_cycle(w, h)
            self.assertIsNotNone(cycle)
            assert cycle is not None
            self.assertEqual(len(cycle), w * h)
            self.assertEqual(len(set(cycle)), w * h)
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_odd_odd_has_no_full_cycle(self) -> None:
        self.assertIsNone(hamiltonian_cycle(5, 5))
        self.assertIsNone(hamiltonian_cycle(7, 9))

    def test_odd_odd_covering_cycle(self) -> None:
        from snake.perfect import covering_cycle

        for w, h in [(5, 5), (7, 7), (9, 9), (5, 7)]:
            cycle, excluded = covering_cycle(w, h)
            self.assertIsNotNone(excluded)
            self.assertEqual(len(cycle), w * h - 1)
            self.assertEqual(len(set(cycle)), w * h - 1)
            self.assertNotIn(excluded, cycle)
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)


class PerfectBotTests(unittest.TestCase):
    def test_fills_small_even_board(self) -> None:
        env = SnakeEnv(6, 6, seed=3, max_idle=0)
        env.reset()
        bot = PerfectBot()
        steps = 0
        while not env.done:
            env.step(bot.act(env))
            steps += 1
            self.assertLess(steps, 10_000, env.death_reason)
        self.assertTrue(env.won, env.death_reason)
        self.assertEqual(env.length, 36)

    def test_fast_hunt_still_fills_and_is_shorter(self) -> None:
        def run(hunt: bool) -> int:
            env = SnakeEnv(8, 8, seed=3, max_idle=0)
            env.reset()
            bot = PerfectBot(hunt=hunt)
            steps = 0
            while not env.done:
                env.step(bot.act(env))
                steps += 1
                self.assertLess(steps, 10_000)
            self.assertTrue(env.won, env.death_reason)
            return steps

        rail = run(False)
        fast = run(True)
        self.assertLessEqual(fast, rail)

    def test_fills_odd_board(self) -> None:
        env = SnakeEnv(5, 5, seed=4, max_idle=0)
        env.reset()
        bot = PerfectBot()
        steps = 0
        while not env.done:
            env.step(bot.act(env))
            steps += 1
            self.assertLess(steps, 10_000, env.death_reason)
        self.assertTrue(env.won, env.death_reason)
        self.assertEqual(env.length, 25)

    def test_hunt_and_rail_win_on_former_failure_seeds(self) -> None:
        cases = [
            (5, 5, 3),
            (6, 6, 3),
            (6, 6, 7),
            (8, 8, 7),
            (20, 12, 7),
        ]
        for w, h, seed in cases:
            for hunt in (False, True):
                env = SnakeEnv(w, h, seed=seed, max_idle=0)
                env.reset()
                bot = PerfectBot(hunt=hunt)
                steps = 0
                while not env.done:
                    env.step(bot.act(env))
                    steps += 1
                    self.assertLess(
                        steps,
                        w * h * 80,
                        f"{w}x{h} hunt={hunt} seed={seed} {env.death_reason} len={env.length}",
                    )
                self.assertTrue(
                    env.won,
                    f"{w}x{h} hunt={hunt} seed={seed} {env.death_reason} len={env.length}",
                )
                self.assertEqual(env.length, w * h)


class PackagedBotTests(unittest.TestCase):
    def test_watcher_modes_fill_without_dying(self) -> None:
        from snake.bot import SnakeBot

        for mode in ("perfect", "neural", "hybrid"):
            bot = SnakeBot(mode=mode)
            env = SnakeEnv(6, 6, seed=3, max_idle=0)
            env.reset()
            steps = 0
            while not env.done:
                env.step(bot.act(env))
                steps += 1
                self.assertLess(steps, 10_000, f"{mode} {env.death_reason}")
            self.assertTrue(env.won, f"{mode} {env.death_reason}")
            self.assertEqual(env.length, 36)


class WatcherPackagingTests(unittest.TestCase):
    def test_watcher_never_starves_neural_mode(self) -> None:
        from snake.web import new_game, STATE

        snap = new_game(8, 8, "neural", seed=3, model_path=None)
        self.assertEqual(snap["mode"], "neural")
        self.assertEqual(STATE["env"].max_idle, 0)

    def test_watcher_rejects_random(self) -> None:
        from snake.web import new_game

        snap = new_game(6, 6, "random", seed=1, model_path=None)
        self.assertEqual(snap["mode"], "perfect")


class NetworkTests(unittest.TestCase):
    def test_untrained_model_knows_the_controls(self) -> None:
        model = SnakeActorCritic()
        view = torch.zeros(2, CHANNELS, VIEW_SIZE, VIEW_SIZE)
        features = torch.zeros(2, N_FEATURES)
        action, log_prob, value = model.act(view, features)
        self.assertEqual(tuple(action.shape), (2,))
        self.assertTrue(torch.all((action >= 0) & (action <= 2)))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertTrue(torch.isfinite(log_prob).all())


if __name__ == "__main__":
    unittest.main()
