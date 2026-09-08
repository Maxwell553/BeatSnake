"""python -m snake <train|play|eval|bot>"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m snake <train|play|watch|eval|record|wins|export>")
        raise SystemExit(2)
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if cmd in {"train", "training"}:
        from snake.train import main as train_main

        train_main()
    elif cmd in {"play"}:
        from snake.play import main as play_main

        play_main()
    elif cmd in {"watch", "web", "view"}:
        from snake.web import main as web_main

        web_main()
    elif cmd in {"eval", "evaluate"}:
        from snake.evaluate import main as eval_main

        eval_main()
    elif cmd in {"record", "animate"}:
        from snake.record import main as record_main

        record_main()
    elif cmd in {"wins", "win-train", "clone"}:
        from snake.win_train import main as win_main

        win_main()
    elif cmd == "export":
        from snake.bot import default_model_path

        print(default_model_path())
    else:
        print(f"Unknown command: {cmd}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
