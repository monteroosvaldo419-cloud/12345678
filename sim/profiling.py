"""Opt-in simulator profiling with no hot-path cost when disabled."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class SimProfiler:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)
        self.ticks = 0

    @contextmanager
    def phase(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.samples[name].append((time.perf_counter() - started) * 1000.0)

    def count(self, name: str, amount: int = 1) -> None:
        self.counts[name] += amount

    def tick(self) -> None:
        self.ticks += 1

    def summary(self) -> dict:
        total = sum(sum(values) for values in self.samples.values())
        phases = {}
        for name, values in sorted(self.samples.items()):
            phases[name] = {
                "avg_ms": sum(values) / len(values) if values else None,
                "max_ms": max(values) if values else None,
                "samples": len(values),
                "percent": (100.0 * sum(values) / total) if total else 0.0,
            }
        return {
            "ticks": self.ticks,
            "phases": phases,
            "counts": dict(self.counts),
            "total_phase_ms": total,
        }
