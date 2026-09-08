"""Theoretically perfect Snake policies that work on any board size.

A neural net trained from scratch can get very strong, but "never die, on any
grid" is a graph problem with a known construction:

  * If at least one dimension is even, the grid has a Hamiltonian cycle. Following
    that cycle is a never-lose strategy once the snake is aligned with it: the
    snake visits every cell and therefore eventually eats every apple.
  * Odd x odd grids have no Hamiltonian cycle. Those boards (and any unaligned
    position) are played with a safe path-to-food / follow-tail search that
    refuses any move which cuts the snake off from its own tail.

`PerfectBot` uses that combination. `HybridBot` lets the trained network choose
a move and falls back to the perfect policy only when that move would be unsafe.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from snake.env import DIRS, SnakeEnv, turn

Pos = Tuple[int, int]


def hamiltonian_cycle(width: int, height: int) -> Optional[List[Pos]]:
    """Return a Hamiltonian cycle, or None when both dimensions are odd."""
    if width < 2 or height < 2:
        return None
    if width % 2 == 0:
        return _cycle_even_width(width, height)
    if height % 2 == 0:
        cycle = _cycle_even_width(height, width)
        return [(y, x) for x, y in cycle]
    return None


def covering_cycle(width: int, height: int) -> Tuple[List[Pos], Optional[Pos]]:
    """Cycle through every cell, or all but one corner on odd x odd boards.

    Grid graphs are bipartite, so an odd x odd board has no Hamiltonian cycle.
    In that case we cycle through N-1 cells and treat the opposite corner as a
    spur that is entered only to eat or to fill the last square.
    """
    cycle = hamiltonian_cycle(width, height)
    if cycle is not None:
        return cycle, None
    return _odd_odd_near_cycle(width, height)


def _odd_odd_near_cycle(width: int, height: int) -> Tuple[List[Pos], Pos]:
    inner = _cycle_even_width(width - 1, height)
    excluded = (width - 1, height - 1)
    spliced: List[Pos] = []
    n = len(inner)
    for i, cell in enumerate(inner):
        spliced.append(cell)
        nxt = inner[(i + 1) % n]
        if (
            cell[0] == width - 2
            and nxt == (width - 2, cell[1] + 1)
            and cell[1] % 2 == 0
            and cell[1] < height - 1
        ):
            spliced.append((width - 1, cell[1]))
            spliced.append((width - 1, cell[1] + 1))
    return spliced, excluded


def _cycle_even_width(width: int, height: int) -> List[Pos]:
    pts: List[Pos] = []
    for x in range(width):
        pts.append((x, 0))
    for i, x in enumerate(range(width - 1, 0, -1)):
        ys: Iterable[int] = range(1, height) if i % 2 == 0 else range(height - 1, 0, -1)
        for y in ys:
            pts.append((x, y))
    for y in range(height - 1, 0, -1):
        pts.append((0, y))
    return pts


def cycle_index_map(cycle: List[Pos]) -> Dict[Pos, int]:
    return {cell: i for i, cell in enumerate(cycle)}


def cycle_dist(a: int, b: int, n: int) -> int:
    return (b - a) % n


def neighbors(pos: Pos, width: int, height: int) -> List[Pos]:
    x, y = pos
    out = []
    for dx, dy in DIRS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            out.append((nx, ny))
    return out


def bfs(start: Pos, goal: Pos, blocked: Set[Pos], width: int, height: int) -> Optional[List[Pos]]:
    if start == goal:
        return []
    open_goal = set(blocked)
    open_goal.discard(goal)
    q = deque([start])
    came: Dict[Pos, Optional[Pos]] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in neighbors(cur, width, height):
            if nxt in came or nxt in open_goal:
                continue
            came[nxt] = cur
            q.append(nxt)
    if goal not in came:
        return None
    path: List[Pos] = []
    cur: Optional[Pos] = goal
    while cur != start:
        if cur is None:
            return None
        path.append(cur)
        cur = came[cur]
    path.reverse()
    return path


def flood_size(start: Pos, blocked: Set[Pos], width: int, height: int) -> int:
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in neighbors(cur, width, height):
            if nxt in seen or nxt in blocked:
                continue
            seen.add(nxt)
            q.append(nxt)
    return len(seen)


def step_body(
    snake: Sequence[Pos],
    nxt: Pos,
    food: Optional[Pos],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[List[Pos]]:
    x, y = nxt
    if width is not None and height is not None:
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
    body = list(snake)
    ate = food is not None and nxt == food
    blocking = body if ate else body[:-1]
    if nxt in blocking:
        return None
    body.insert(0, nxt)
    if not ate:
        body.pop()
    return body


def simulate_path(
    snake: Sequence[Pos],
    food: Optional[Pos],
    path: Sequence[Pos],
) -> Optional[List[Pos]]:
    body = list(snake)
    for i, cell in enumerate(path):
        nxt = step_body(body, cell, food, width=None, height=None)
        if nxt is None:
            return None
        body = nxt
        if food is not None and cell == food and i != len(path) - 1:
            return None
    return body


def can_reach_tail(body: Sequence[Pos], width: int, height: int) -> bool:
    if len(body) <= 1:
        return True
    blocked = set(body[:-1])
    return bfs(body[0], body[-1], blocked, width, height) is not None


def empty_cells(body: Sequence[Pos], width: int, height: int) -> List[Pos]:
    occupied = set(body)
    return [(x, y) for y in range(height) for x in range(width) if (x, y) not in occupied]


def remaining_space_connected(body: Sequence[Pos], width: int, height: int) -> bool:
    """True if every free cell is reachable from the head's empty neighbors.

    That is the endgame win condition: food will respawn in the same pocket the
    head already occupies, so the snake can finish the board without the tail.
    """
    occupied = set(body)
    area = width * height
    remaining = area - len(body)
    if remaining <= 0:
        return True
    starts = [n for n in neighbors(body[0], width, height) if n not in occupied]
    if not starts:
        return remaining == 0
    seen = set(starts)
    q = deque(starts)
    while q:
        cur = q.popleft()
        for nxt in neighbors(cur, width, height):
            if nxt in seen or nxt in occupied:
                continue
            seen.add(nxt)
            q.append(nxt)
    return len(seen) == remaining


class PerfectBot:
    """Never-die policy. `hunt=True` takes the shortest safe cycle shortcut to food.

    The covering cycle is the safety rail: following it forever cannot die, and it
    visits every cell so the snake always fills the board. Shortcuts are only
    taken when the cycle path from the new head to the tail is still clear, so a
    sequence of hunts cannot cut the snake off from its own tail.
    """

    def __init__(self, hunt: bool = True) -> None:
        self.hunt = hunt
        self._cycle: Optional[List[Pos]] = None
        self._index: Dict[Pos, int] = {}
        self._excluded: Optional[Pos] = None
        self._size: Tuple[int, int] = (0, 0)

    def _ensure_cycle(self, env: SnakeEnv) -> None:
        size = (env.width, env.height)
        if size == self._size:
            return
        self._size = size
        self._cycle, self._excluded = covering_cycle(env.width, env.height)
        self._index = cycle_index_map(self._cycle)

    def act(self, env: SnakeEnv) -> int:
        self._ensure_cycle(env)
        if self._excluded is not None and env.head == self._excluded:
            return self._leave_excluded(env)

        need_excluded = self._should_enter_excluded(env)
        if need_excluded:
            enter = self._enter_excluded_action(env)
            if enter is not None:
                return enter

        action = self._hamiltonian_action(env)
        if action is not None:
            return action
        # Unconstrained hunt-to-food is how the "perfect" bot used to die.
        # If the cycle has no legal step, stall with a tail-reachable move.
        return self._stall_action(env)

    def _should_enter_excluded(self, env: SnakeEnv) -> bool:
        if self._excluded is None:
            return False
        if self._excluded in env.snake:
            return False
        if env.food == self._excluded:
            return True
        return env.length >= env.area - 1

    def _enter_excluded_action(self, env: SnakeEnv) -> Optional[int]:
        assert self._excluded is not None
        if self.is_immediate_death(env, self._excluded):
            return None
        action = self.direction_to_action(env, self._excluded)
        if action is None:
            return None
        body = step_body(env.snake, self._excluded, env.food, env.width, env.height)
        if body is None:
            return None
        if len(body) >= env.area:
            return action
        new_dir = turn(env.direction, action)
        if self._first_legal_from(body, new_dir, env.width, env.height) is None:
            return None
        return action

    def _leave_excluded(self, env: SnakeEnv) -> int:
        for dest in neighbors(env.head, env.width, env.height):
            if dest in self._index and not self.is_immediate_death(env, dest):
                action = self.direction_to_action(env, dest)
                if action is not None:
                    return action
        return self._first_legal(env)

    def _first_legal(self, env: SnakeEnv) -> int:
        action = self._first_legal_from(env.snake, env.direction, env.width, env.height)
        return action if action is not None else 1

    def _first_legal_from(
        self, body: Sequence[Pos], direction: int, width: int, height: int
    ) -> Optional[int]:
        blocking = set(body[:-1])
        hx, hy = body[0]
        for action in (0, 1, 2):
            nd = turn(direction, action)
            nx, ny = hx + DIRS[nd][0], hy + DIRS[nd][1]
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if (nx, ny) in blocking:
                continue
            return action
        return None

    def _adjacent_food_action(self, env: SnakeEnv) -> Optional[int]:
        if env.food is None:
            return None
        for action in (1, 0, 2):
            nd = turn(env.direction, action)
            nxt = (env.head[0] + DIRS[nd][0], env.head[1] + DIRS[nd][1])
            if nxt == env.food and not self.is_immediate_death(env, nxt):
                return action
        return None

    def _can_resume_cycle(self, body: Sequence[Pos]) -> bool:
        if not self._cycle or not body:
            return False
        head = body[0]
        if head not in self._index:
            return False
        nxt = self._cycle[(self._index[head] + 1) % len(self._cycle)]
        return nxt not in set(body[:-1])

    def _safe_after(self, body: Sequence[Pos], env: SnakeEnv) -> bool:
        if len(body) >= env.area:
            return True
        if not can_reach_tail(body, env.width, env.height):
            return False
        return self._can_resume_cycle(body)

    def _hamiltonian_action(self, env: SnakeEnv) -> Optional[int]:
        assert self._cycle is not None
        n = len(self._cycle)
        head = env.head
        if head not in self._index:
            return None
        h = self._index[head]
        food = env.food if env.food in self._index else None
        default_next = self._cycle[(h + 1) % n]
        choice = (
            default_next
            if not self.is_immediate_death(env, default_next)
            and self.direction_to_action(env, default_next) is not None
            else None
        )

        if self.hunt and food is not None:
            f = self._index[food]
            dist_food = cycle_dist(h, f, n)
            best_left = dist_food + 1
            if choice is not None:
                best_left = cycle_dist(self._index[choice], f, n)
            for action in (0, 1, 2):
                nd = turn(env.direction, action)
                nxt = (head[0] + DIRS[nd][0], head[1] + DIRS[nd][1])
                if not self._cycle_move_ok(env, nxt):
                    continue
                ni = self._index[nxt]
                jump = cycle_dist(h, ni, n)
                if jump == 0 or jump > dist_food:
                    continue
                left = cycle_dist(ni, f, n)
                if left < best_left:
                    best_left = left
                    choice = nxt

        if choice is None or self.is_immediate_death(env, choice):
            return None
        return self.direction_to_action(env, choice)

    def _cycle_move_ok(self, env: SnakeEnv, nxt: Pos) -> bool:
        """True if this cycle cell is a legal step that can still follow the rail."""
        if nxt not in self._index:
            return False
        if self.direction_to_action(env, nxt) is None:
            return False
        if self.is_immediate_death(env, nxt):
            return False
        body = step_body(env.snake, nxt, env.food, env.width, env.height)
        if body is None:
            return False
        if len(body) >= env.area:
            return True
        tail = body[-1]
        if tail not in self._index:
            return can_reach_tail(body, env.width, env.height)
        room = cycle_dist(self._index[body[0]], self._index[tail], len(self._cycle))
        if room <= len(body):
            return False
        return self._cycle_path_clear(body)

    def _cycle_path_clear(self, body: Sequence[Pos]) -> bool:
        """Walking the covering cycle from head, the first body cell hit is the tail."""
        n = len(self._cycle)
        head = body[0]
        tail = body[-1]
        if head not in self._index or tail not in self._index:
            return False
        occupied = set(body)
        i = self._index[head]
        for _ in range(n):
            i = (i + 1) % n
            cell = self._cycle[i]
            if cell == tail:
                return True
            if cell in occupied:
                return False
        return False

    def direction_to_action(self, env: SnakeEnv, dest: Pos) -> Optional[int]:
        hx, hy = env.head
        want = (dest[0] - hx, dest[1] - hy)
        for action in (0, 1, 2):
            nd = turn(env.direction, action)
            if DIRS[nd] == want:
                return action
        return None

    def _best_safe_eat_action(self, env: SnakeEnv) -> Optional[int]:
        food = env.food
        if food is None:
            return None
        best_action: Optional[int] = None
        best_remaining = 10**9
        for action in (1, 0, 2):
            nd = turn(env.direction, action)
            nxt = (env.head[0] + DIRS[nd][0], env.head[1] + DIRS[nd][1])
            body = step_body(env.snake, nxt, food, env.width, env.height)
            if body is None:
                continue
            if nxt == food:
                if self._safe_after(body, env):
                    return action
                continue
            path = bfs(body[0], food, set(body[:-1]), env.width, env.height)
            if not path:
                continue
            virtual = simulate_path(body, food, path)
            if virtual is None:
                continue
            if self._safe_after(virtual, env):
                if len(path) < best_remaining:
                    best_remaining = len(path)
                    best_action = action
        return best_action

    def _stall_action(self, env: SnakeEnv) -> int:
        food = env.food
        ranked: List[Tuple[tuple, int]] = []
        for action in (0, 1, 2):
            nd = turn(env.direction, action)
            nxt = (env.head[0] + DIRS[nd][0], env.head[1] + DIRS[nd][1])
            body = step_body(env.snake, nxt, food, env.width, env.height)
            if body is None:
                continue
            ate = food is not None and nxt == food
            if ate and not (
                len(body) >= env.area
                or can_reach_tail(body, env.width, env.height)
                or remaining_space_connected(body, env.width, env.height)
            ):
                continue
            if not (
                can_reach_tail(body, env.width, env.height)
                or remaining_space_connected(body, env.width, env.height)
            ):
                continue
            blocked = set(body[:-1])
            food_reach = 0
            man = 0
            if food is not None:
                food_reach = 1 if bfs(body[0], food, blocked, env.width, env.height) is not None else 0
                man = -(abs(body[0][0] - food[0]) + abs(body[0][1] - food[1]))
            tail_path = bfs(body[0], body[-1], blocked, env.width, env.height) or []
            space = flood_size(body[0], blocked, env.width, env.height)
            key = (0 if ate else 1, food_reach, len(tail_path), man, space)
            ranked.append((key, action))
        if ranked:
            ranked.sort(reverse=True)
            return ranked[0][1]
        return self._longest_safe(env)

    def _longest_safe(self, env: SnakeEnv) -> int:
        best_action = None
        best_space = -1
        fallback = None
        blocked = set(env.snake[:-1])
        for action in (0, 1, 2):
            nd = turn(env.direction, action)
            nxt = (env.head[0] + DIRS[nd][0], env.head[1] + DIRS[nd][1])
            if self.is_immediate_death(env, nxt):
                continue
            fallback = action if fallback is None else fallback
            if env.food is not None and nxt == env.food:
                continue
            space = flood_size(nxt, blocked, env.width, env.height)
            if space > best_space:
                best_space = space
                best_action = action
        if best_action is not None:
            return best_action
        forced = self._adjacent_food_action(env)
        if forced is not None:
            return forced
        return fallback if fallback is not None else self._first_legal(env)

    def is_immediate_death(self, env: SnakeEnv, nxt: Pos) -> bool:
        x, y = nxt
        if x < 0 or y < 0 or x >= env.width or y >= env.height:
            return True
        return nxt in set(env.snake[:-1])


class HybridBot:
    """Neural policy with a perfect safety filter."""

    def __init__(self, neural_act, perfect: Optional[PerfectBot] = None) -> None:
        self.neural_act = neural_act
        self.perfect = perfect or PerfectBot()

    def act(self, env: SnakeEnv) -> int:
        proposed = int(self.neural_act(env))
        if self._is_safe_action(env, proposed):
            return proposed
        eat = self.perfect._best_safe_eat_action(env)
        if eat is not None:
            return eat
        return self.perfect._stall_action(env)

    def _is_safe_action(self, env: SnakeEnv, action: int) -> bool:
        nd = turn(env.direction, action)
        nxt = (env.head[0] + DIRS[nd][0], env.head[1] + DIRS[nd][1])
        if self.perfect.is_immediate_death(env, nxt):
            return False
        body = step_body(env.snake, nxt, env.food, env.width, env.height)
        if body is None:
            return False
        if len(body) >= env.area:
            return True
        return can_reach_tail(body, env.width, env.height) or remaining_space_connected(
            body, env.width, env.height
        )
