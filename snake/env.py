"""Classic Snake environment with a size-invariant observation space.

The agent never receives a raw board of a fixed size. Every observation is:
  1. An 11x11 egocentric crop, rotated so the snake always faces "up".
  2. A small vector of board-normalized scalars (food bearing, length ratio, etc.).

That is enough for the same network to play 5x5, 8x12, 20x20, or any other grid
without retraining the architecture. Controls are relative: left / straight / right.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Relative actions: 0 = turn left, 1 = continue straight, 2 = turn right.
N_ACTIONS = 3
VIEW_SIZE = 11
CHANNELS = 3  # body, food, walls
# 13 board/control scalars + 11 covering-cycle scalars (see observe()).
N_FEATURES = 27

# World directions: 0=up, 1=right, 2=down, 3=left. y increases downward.
DIRS: Tuple[Tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


def turn(direction: int, action: int) -> int:
    """Apply a relative action to a heading."""
    return (direction + action - 1) % 4


class SnakeEnv:
    """Single Snake game. Coordinates are (x, y) with origin at top-left."""

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        max_idle: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if width < 4 or height < 4:
            raise ValueError("Board must be at least 4x4 so a length-3 snake can spawn.")
        self.width = int(width)
        self.height = int(height)
        self.area = self.width * self.height
        # None = 2 * area (training default). 0 = never starve.
        self._idle_override = max_idle
        self.max_idle = self._compute_max_idle()
        self.rng = np.random.default_rng(seed)

        self.snake: List[Tuple[int, int]] = []
        self.direction = 1
        self.food: Optional[Tuple[int, int]] = None
        self.steps = 0
        self.idle = 0
        self.done = False
        self.won = False
        self.death_reason = ""

    @property
    def length(self) -> int:
        return len(self.snake)

    @property
    def head(self) -> Tuple[int, int]:
        return self.snake[0]

    def set_size(self, width: int, height: int) -> None:
        if width < 4 or height < 4:
            raise ValueError("Board must be at least 4x4.")
        self.width = int(width)
        self.height = int(height)
        self.area = self.width * self.height
        self.max_idle = self._compute_max_idle()

    def _compute_max_idle(self) -> int:
        if self._idle_override is None:
            return self.area * 2
        return int(self._idle_override)

    def reset(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if width is not None or height is not None:
            self.set_size(width or self.width, height or self.height)

        self.direction = int(self.rng.integers(0, 4))
        dx, dy = DIRS[self.direction]
        cx, cy = self.width // 2, self.height // 2
        self.snake = [(cx - dx * i, cy - dy * i) for i in range(3)]
        self.food = None
        self._place_food()
        self.steps = 0
        self.idle = 0
        self.done = False
        self.won = False
        self.death_reason = ""
        return self.observe()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called after the episode ended; call reset().")

        action = int(action)
        if action not in (0, 1, 2):
            raise ValueError(f"action must be 0, 1, or 2; got {action}")

        self.direction = turn(self.direction, action)
        dx, dy = DIRS[self.direction]
        hx, hy = self.head
        nx, ny = hx + dx, hy + dy
        self.steps += 1
        self.idle += 1

        reward = 0.0
        info: Dict[str, Any] = {"ate": False, "won": False, "reason": ""}

        if nx < 0 or ny < 0 or nx >= self.width or ny >= self.height:
            return self._die(-1.0, "wall", info)

        body_without_tail = self.snake[:-1]
        if (nx, ny) in body_without_tail:
            return self._die(-1.0, "self", info)

        ate = self.food is not None and (nx, ny) == self.food
        self.snake.insert(0, (nx, ny))
        if ate:
            reward = 1.0
            info["ate"] = True
            self.idle = 0
            if len(self.snake) >= self.area:
                self.done = True
                self.won = True
                self.food = None
                reward += 5.0
                info["won"] = True
                info["reason"] = "win"
                info["length"] = self.length
                info["steps"] = self.steps
                return self.observe(), reward, True, info
            self._place_food()
        else:
            self.snake.pop()

        if self.max_idle > 0 and self.idle >= self.max_idle:
            return self._die(-1.0, "starve", info)

        info["length"] = self.length
        info["steps"] = self.steps
        return self.observe(), reward, False, info

    def observe(self) -> Dict[str, np.ndarray]:
        view = np.zeros((CHANNELS, VIEW_SIZE, VIEW_SIZE), dtype=np.float32)
        features = np.zeros((N_FEATURES,), dtype=np.float32)
        if not self.snake:
            return {"view": view, "features": features}

        hx, hy = self.head
        body_index = {pos: i for i, pos in enumerate(self.snake)}
        half = VIEW_SIZE // 2

        for row in range(VIEW_SIZE):
            for col in range(VIEW_SIZE):
                dx_e = col - half
                dy_e = row - half
                wx, wy = self._ego_to_world(dx_e, dy_e)
                if wx < 0 or wy < 0 or wx >= self.width or wy >= self.height:
                    view[2, row, col] = 1.0
                    continue
                idx = body_index.get((wx, wy))
                if idx is not None:
                    # Head is 1.0; tail is the smallest positive value.
                    view[0, row, col] = 1.0 - idx / max(self.length, 1)
                if self.food == (wx, wy):
                    view[1, row, col] = 1.0

        fx, fy = self.food if self.food is not None else self.head
        food_dx, food_dy = self._world_to_ego(fx - hx, fy - hy)
        max_side = float(max(self.width, self.height))
        features[0] = -food_dy / max_side  # ahead
        features[1] = food_dx / max_side  # right
        features[2] = (abs(fx - hx) + abs(fy - hy)) / float(self.width + self.height)
        features[3] = self.length / float(self.area)
        features[4] = self.idle / float(self.max_idle) if self.max_idle > 0 else 0.0
        features[5] = 1.0 if self._danger(0) else 0.0
        features[6] = 1.0 if self._danger(1) else 0.0
        features[7] = 1.0 if self._danger(2) else 0.0
        features[8] = self._wall_distance(0) / max_side
        features[9] = self._wall_distance(1) / max_side
        features[10] = self._wall_distance(2) / max_side
        features[11] = 1.0 if self.food is not None else 0.0
        features[12] = float(self.direction) / 3.0
        self._fill_cycle_features(features)
        return {"view": view, "features": features}

    def _fill_cycle_features(self, features: np.ndarray) -> None:
        """Append covering-cycle scalars so a net can imitate PerfectBot."""
        from snake.perfect import covering_cycle, cycle_dist, cycle_index_map

        cycle, _excluded = covering_cycle(self.width, self.height)
        index = cycle_index_map(cycle)
        n = len(cycle)
        head = self.head
        if head not in index:
            return
        h = index[head]
        food = self.food if self.food in index else None
        dist_food = cycle_dist(h, index[food], n) / n if food is not None else 1.0
        t = index.get(self.snake[-1])
        room = cycle_dist(h, t, n) / n if t is not None else 0.0
        features[13] = dist_food
        features[14] = room
        blocking = set(self.snake[:-1])
        raw_left = []
        for action in (0, 1, 2):
            nd = turn(self.direction, action)
            nxt = (head[0] + DIRS[nd][0], head[1] + DIRS[nd][1])
            base = 15 + action * 3
            if nxt not in index or nxt in blocking:
                features[base] = 0.0
                features[base + 1] = 1.0
                features[base + 2] = 0.0
                raw_left.append(None)
                continue
            ni = index[nxt]
            jump = cycle_dist(h, ni, n)
            left = cycle_dist(ni, index[food], n) / n if food is not None else 1.0
            room_n = cycle_dist(ni, t, n) if t is not None else 0
            need = self.length if food is not None and nxt == food else max(self.length - 1, 1)
            shortcut = room_n > need and 0 < jump <= (
                cycle_dist(h, index[food], n) if food is not None else n
            )
            rail = jump == 1
            features[base] = 1.0
            features[base + 1] = left
            features[base + 2] = 1.0 if shortcut else 0.0
            raw_left.append(left if (rail or shortcut) else None)
        cand = [v for v in raw_left if v is not None]
        min_left = min(cand) if cand else 1.0
        for action, left in enumerate(raw_left):
            features[24 + action] = (left - min_left) if left is not None else 1.0

    def occupancy(self) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=np.int32)
        for i, (x, y) in enumerate(self.snake):
            grid[y, x] = self.length - i
        return grid

    def clone_state(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "snake": list(self.snake),
            "direction": self.direction,
            "food": self.food,
            "steps": self.steps,
            "idle": self.idle,
            "done": self.done,
            "won": self.won,
            "death_reason": self.death_reason,
            "max_idle": self.max_idle,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.width = state["width"]
        self.height = state["height"]
        self.area = self.width * self.height
        self.snake = list(state["snake"])
        self.direction = state["direction"]
        self.food = state["food"]
        self.steps = state["steps"]
        self.idle = state["idle"]
        self.done = state["done"]
        self.won = state["won"]
        self.death_reason = state["death_reason"]
        self.max_idle = state["max_idle"]

    def _die(
        self, reward: float, reason: str, info: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.done = True
        self.death_reason = reason
        info["reason"] = reason
        info["length"] = self.length
        info["steps"] = self.steps
        return self.observe(), reward, True, info

    def _place_food(self) -> None:
        occupied = set(self.snake)
        empty = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if not empty:
            self.food = None
            return
        choice = int(self.rng.integers(0, len(empty)))
        self.food = empty[choice]

    def _danger(self, action: int) -> bool:
        nd = turn(self.direction, action)
        dx, dy = DIRS[nd]
        nx, ny = self.head[0] + dx, self.head[1] + dy
        if nx < 0 or ny < 0 or nx >= self.width or ny >= self.height:
            return True
        blocking = self.snake[:-1]
        return (nx, ny) in blocking

    def _wall_distance(self, action: int) -> float:
        nd = turn(self.direction, action)
        dx, dy = DIRS[nd]
        x, y = self.head
        dist = 0
        while True:
            x += dx
            y += dy
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                break
            dist += 1
        return float(dist)

    def _world_to_ego(self, dx: int, dy: int) -> Tuple[int, int]:
        d = self.direction
        if d == 0:
            return dx, dy
        if d == 1:
            return dy, -dx
        if d == 2:
            return -dx, -dy
        return -dy, dx

    def _ego_to_world(self, dx_e: int, dy_e: int) -> Tuple[int, int]:
        hx, hy = self.head
        d = self.direction
        if d == 0:
            dx, dy = dx_e, dy_e
        elif d == 1:
            dx, dy = -dy_e, dx_e
        elif d == 2:
            dx, dy = -dx_e, -dy_e
        else:
            dx, dy = dy_e, -dx_e
        return hx + dx, hy + dy


class VecSnakeEnv:
    """Independent Snake games that share one observation layout (any mix of sizes)."""

    def __init__(
        self,
        num_envs: int,
        sizes: Sequence[Tuple[int, int]],
        seed: int = 0,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if not sizes:
            raise ValueError("sizes must be non-empty")
        self.num_envs = int(num_envs)
        self.sizes = [(int(w), int(h)) for w, h in sizes]
        self.rng = np.random.default_rng(seed)
        self.envs = [
            SnakeEnv(width=self.sizes[0][0], height=self.sizes[0][1], seed=seed + i)
            for i in range(self.num_envs)
        ]

    def _sample_size(self) -> Tuple[int, int]:
        idx = int(self.rng.integers(0, len(self.sizes)))
        return self.sizes[idx]

    def reset(self) -> Dict[str, np.ndarray]:
        views = []
        feats = []
        for env in self.envs:
            w, h = self._sample_size()
            obs = env.reset(width=w, height=h)
            views.append(obs["view"])
            feats.append(obs["features"])
        return {"view": np.stack(views), "features": np.stack(feats)}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        views = []
        feats = []
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=np.bool_)
        infos: List[Dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(int(actions[i]))
            info = dict(info)
            info["width"] = env.width
            info["height"] = env.height
            if done:
                info["terminal_length"] = env.length
                info["terminal_reason"] = info.get("reason", env.death_reason)
                info["won"] = env.won
                w, h = self._sample_size()
                obs = env.reset(width=w, height=h)
            views.append(obs["view"])
            feats.append(obs["features"])
            rewards[i] = reward
            dones[i] = done
            infos.append(info)
        return {"view": np.stack(views), "features": np.stack(feats)}, rewards, dones, infos
