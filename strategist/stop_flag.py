"""Cooperative stop for long detached runs.

The runs are started detached (setsid nohup, tmux), so stopping them by pid
needs a person at a shell. Instead each run watches a stop file: when the file
appears (created after the run started), the run terminates its child
processes (Codex CLI calls, OpenHands subprocesses) and exits. Every run is
resumable, so a stop loses at most the work in flight.

    STRATEGIST_STOP_FILE=runs/STOP python strategist/search.py ...   # the run watches this path
    touch runs/STOP                                                 # stops it
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def install(path: str | os.PathLike | None, label: str = "", interval: float = 10.0) -> None:
    """Start a daemon thread that shuts this process down when `path` appears.
    A stop file older than this process is ignored (a leftover from an earlier
    stop), so a fresh run never exits on a stale flag."""
    if not path:
        return
    stop = Path(path)
    started = time.time()

    def _watch() -> None:
        while True:
            time.sleep(interval)
            try:
                fresh = stop.exists() and stop.stat().st_mtime >= started - 1
            except OSError:
                fresh = False
            if fresh:
                print(f"[stop] {stop} found — {label or 'run'} shutting down", flush=True)
                shutdown()

    threading.Thread(target=_watch, name="stop-flag", daemon=True).start()


def shutdown(grace: float = 10.0) -> None:
    """Terminate every descendant process, then exit this one."""
    try:
        import psutil

        kids = psutil.Process().children(recursive=True)
        for k in kids:
            try:
                k.terminate()
            except Exception:
                pass
        _, alive = psutil.wait_procs(kids, timeout=grace)
        for k in alive:
            try:
                k.kill()
            except Exception:
                pass
    except Exception:
        pass
    os._exit(3)
