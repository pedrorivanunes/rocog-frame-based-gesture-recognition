"""Reduce repeated timings to what a latency table reports.

Its own module, with nothing imported beyond the standard library, because the
things that need timing do not all live in one environment. The networks are
timed where PyTorch is installed; the pose detector runs in an environment of
its own that has no PyTorch at all, and the classical optical flow methods need
neither. Each of them reports the same three figures in the same table, and the
arithmetic that produces those figures should exist once rather than once per
environment.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

WARMUP = 5
REPEATS = 30


@dataclass(frozen=True)
class Summary:
    """What a set of timings is reported as.

    The median and the interquartile range rather than a mean and a standard
    deviation, because wall clock on a general purpose machine is bounded below
    and has a long right tail: one descheduled run drags a mean and leaves a
    median alone. The minimum is kept as the closest thing to an uninterrupted
    run.
    """

    median_ms: float
    iqr_ms: float
    min_ms: float
    samples: int


def summarise(samples: Sequence[float]) -> Summary:
    """Reduce raw timings, in milliseconds, to what gets reported.

    Args:
        samples: One duration per timed repetition, in milliseconds.

    Returns:
        The summary described by ``Summary``.

    Raises:
        ValueError: If there is nothing to summarise. An empty set here means
            the timing loop never ran, which is a bug rather than a result.
    """
    if not samples:
        raise ValueError("no timings to summarise")

    ordered = sorted(samples)
    if len(ordered) >= 4:
        quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
        iqr = quartiles[2] - quartiles[0]
    else:
        iqr = ordered[-1] - ordered[0]

    return Summary(statistics.median(ordered), iqr, ordered[0], len(ordered))


def time_repeatedly(
    call: Callable[[], object], warmup: int = WARMUP, repeats: int = REPEATS
) -> Summary:
    """Run something repeatedly and report how long it took, warmup excluded.

    The warmup is not padding: the first calls pay for allocation, caches and
    whatever a library initialises lazily, which is a cost paid once and not
    per decision. A call that hands work to an accelerator has to wait for it
    inside ``call`` itself, or the clock measures how fast a queue accepts work.

    Args:
        call: What one repetition does.
        warmup: Calls made first and not timed.
        repeats: Calls timed.

    Returns:
        The timed repetitions, summarised.
    """
    for _ in range(warmup):
        call()

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000)

    return summarise(samples)
