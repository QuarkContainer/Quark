"""Aggregate statistics for benchmark samples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricStats:
    samples: int
    mean: float
    p50: float
    min: float
    max: float
    unit: str

    @classmethod
    def from_values(cls, values: list[float], unit: str) -> MetricStats:
        if not values:
            return cls(0, 0, 0, 0, 0, unit)
        s = sorted(values)
        n = len(s)
        p50 = s[n // 2]
        return cls(
            samples=n,
            mean=round(sum(s) / n, 2),
            p50=round(p50, 2),
            min=round(s[0], 2),
            max=round(s[-1], 2),
            unit=unit,
        )
