"""Browser viewer: pick a board size and watch a bot play."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from snake.bot import SnakeBot
from snake.env import SnakeEnv

PAGE = Path(__file__).with_name("watch.html")
LOCK = threading.Lock()
STATE = {
    "env": None,
    "bot": None,
    "mode": "perfect",
    "model_path": None,
}


def snapshot(env: SnakeEnv, mode: str) -> dict:
    return {
        "width": env.width,
        "height": env.height,
        "snake": list(env.snake),
        "food": env.food,
        "direction": env.direction,
        "length": env.length,
        "area": env.area,
        "steps": env.steps,
        "done": env.done,
        "won": env.won,
        "reason": env.death_reason,
        "mode": mode,
    }


def new_game(width: int, height: int, mode: str, seed: int, model_path: Optional[str]) -> dict:
    width = max(4, min(40, int(width)))
    height = max(4, min(40, int(height)))
    if mode not in {"neural", "perfect", "hybrid"}:
        mode = "perfect"
    idle = 0
    env = SnakeEnv(width, height, seed=seed, max_idle=idle)
    env.reset()
    bot = SnakeBot(mode=mode, model_path=model_path)
    STATE["env"] = env
    STATE["bot"] = bot
    STATE["mode"] = mode
    return snapshot(env, mode)


def step_game() -> dict:
    env: SnakeEnv = STATE["env"]
    bot: SnakeBot = STATE["bot"]
    if env is None or bot is None:
        raise RuntimeError("No game. Call /api/new first.")
    if not env.done:
        action = bot.act(env)
        env.step(action)
    return snapshot(env, STATE["mode"])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _json(self, payload: dict, code: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html(self) -> None:
        html = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._html()
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            body = {}
        path = urlparse(self.path).path
        try:
            with LOCK:
                if path == "/api/new":
                    payload = new_game(
                        body.get("width", 12),
                        body.get("height", 12),
                        body.get("mode", "perfect"),
                        int(body.get("seed", 0)),
                        STATE["model_path"],
                    )
                    self._json(payload)
                    return
                if path == "/api/step":
                    self._json(step_game())
                    return
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc)}, 500)
            return
        self._json({"error": "not found"}, 404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a browser board and watch a Snake bot.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    STATE["model_path"] = args.model or None
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Snake viewer at {url}", flush=True)
    print("Pick a board size in the browser, then press Play.", flush=True)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
