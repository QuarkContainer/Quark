"""Benchmark statistics and reports."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LatencyStats:
    samples: int
    mean: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float

    @classmethod
    def from_values(cls, values: list[float]) -> LatencyStats:
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0)
        s = sorted(values)
        n = len(s)

        def pct(p: float) -> float:
            idx = min(n - 1, max(0, int(math.ceil(p / 100.0 * n) - 1)))
            return round(s[idx], 2)

        return cls(
            samples=n,
            mean=round(statistics.mean(s), 2),
            p50=pct(50),
            p95=pct(95),
            p99=pct(99),
            min=round(s[0], 2),
            max=round(s[-1], 2),
        )


@dataclass
class BenchReport:
    backend: str
    profile: str
    host: str
    timestamp: str
    stats: LatencyStats
    success_rate: float
    raw_ms: list[float] = field(repr=False)
    errors: list[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["stats"] = asdict(self.stats)
        return d

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return path

    @classmethod
    def build(
        cls,
        *,
        backend: str,
        profile: str,
        host: str,
        samples: list[float],
        errors: list[str],
        notes: str = "",
    ) -> BenchReport:
        ok = len(samples)
        total = ok + len(errors)
        rate = ok / total if total else 0.0
        return cls(
            backend=backend,
            profile=profile,
            host=host,
            timestamp=datetime.now(timezone.utc).isoformat(),
            stats=LatencyStats.from_values(samples),
            success_rate=rate,
            raw_ms=samples,
            errors=errors,
            notes=notes,
        )
