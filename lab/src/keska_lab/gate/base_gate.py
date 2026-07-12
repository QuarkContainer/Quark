"""Gate result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    gate: str
    ok: bool
    message: str
    duration_s: float = 0.0
