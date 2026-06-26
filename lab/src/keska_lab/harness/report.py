"""Structured benchmark report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from keska_lab.harness.stats import MetricStats


@dataclass
class SuiteReport:
    backend: str
    host: str
    timestamp: str
    suite: str
    workload: str
    iterations: int
    image: str
    metrics: dict[str, MetricStats | float | str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": "2.0",
            "backend": self.backend,
            "host": self.host,
            "timestamp": self.timestamp,
            "suite": self.suite,
            "workload": self.workload,
            "iterations": self.iterations,
            "image": self.image,
            "metrics": {},
            "errors": self.errors,
            "notes": self.notes,
            "skipped": self.skipped,
        }
        for k, v in self.metrics.items():
            if isinstance(v, MetricStats):
                out["metrics"][k] = asdict(v)
            else:
                out["metrics"][k] = v
        return out

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return path

    @classmethod
    def new(
        cls,
        *,
        backend: str,
        host: str,
        suite: str,
        workload: str,
        image: str,
        iterations: int,
    ) -> SuiteReport:
        return cls(
            backend=backend,
            host=host,
            timestamp=datetime.now(timezone.utc).isoformat(),
            suite=suite,
            workload=workload,
            iterations=iterations,
            image=image,
        )
