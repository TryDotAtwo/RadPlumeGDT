from __future__ import annotations

import sys
import time
from threading import Event, Thread
from typing import Callable, TypeVar

from tqdm import tqdm


T = TypeVar("T")


def run_with_tqdm_heartbeat(description: str, operation: Callable[[], T]) -> T:
    done = Event()
    state: dict[str, object] = {}

    def _runner() -> None:
        try:
            state["result"] = operation()
        except Exception as exc:  # pragma: no cover - passthrough helper
            state["error"] = exc
        finally:
            done.set()

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    if sys.stdout.isatty():
        progress = tqdm(
            desc=description,
            total=None,
            unit="tick",
            dynamic_ncols=True,
            ascii=True,
            mininterval=1.0,
            leave=True,
        )
        try:
            while not done.wait(5.0):
                progress.update(1)
            progress.update(1)
        finally:
            progress.close()
    else:
        start = time.monotonic()
        print(f"{description}: started")
        while not done.wait(30.0):
            elapsed = int(time.monotonic() - start)
            print(f"{description}: still running... {elapsed // 60}m {elapsed % 60:02d}s elapsed")
        elapsed = int(time.monotonic() - start)
        print(f"{description}: finished after {elapsed // 60}m {elapsed % 60:02d}s")

    if "error" in state:
        raise state["error"]  # type: ignore[misc]
    return state["result"]  # type: ignore[return-value]
