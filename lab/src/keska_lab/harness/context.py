"""Runtime context passed to benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from keska_lab.harness.workload import WorkloadSpec

if TYPE_CHECKING:
    from keska_lab.backends.base import SandboxBackend


@dataclass
class CaseContext:
    backend: SandboxBackend
    workload: WorkloadSpec
    verbose: bool = True
    extras: dict = field(default_factory=dict)
    _pause_resume_cache: tuple[float, float, float] | None = field(
        default=None, repr=False
    )
