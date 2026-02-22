"""Timing context manager for benchmarking."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    """Stores elapsed time from a timing context."""

    elapsed: float = 0.0


@contextmanager
def timer():
    """Context manager that measures wall-clock time in seconds.

    Usage:
        t = TimingResult()
        with timer() as t:
            do_something()
        print(f"Took {t.elapsed:.3f}s")
    """
    result = TimingResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed = time.perf_counter() - start
