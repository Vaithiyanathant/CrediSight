"""Small timing helpers used by the pipelines and the API middleware."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC


class Timer:
    __slots__ = ("_start", "elapsed_ms")

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.elapsed_ms = 0.0

    def stop(self) -> float:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        return self.elapsed_ms


@contextmanager
def timed(sink: dict, key: str = "elapsed_ms") -> Iterator[Timer]:
    t = Timer()
    try:
        yield t
    finally:
        sink[key] = round(t.stop(), 3)


def utcnow_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
