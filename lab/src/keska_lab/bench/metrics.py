"""Extended benchmark metrics — delegates to harness runner."""

from __future__ import annotations

from keska_lab.backends.base import SandboxBackend
from keska_lab.harness import MetricStats, SuiteReport, run_suite

# Backward compatibility aliases
FullBenchReport = SuiteReport


def run_full_suite(
    backend: SandboxBackend,
    *,
    n: int = 10,
    image: str = "busybox",
    verbose: bool = True,
    suite: str = "light",
    workload: str | None = None,
) -> SuiteReport:
    """Run a benchmark suite (default: light / legacy "full")."""
    wl = workload or _image_to_workload(image)
    return run_suite(backend, suite=suite, workload=wl, n=n, verbose=verbose)


def _image_to_workload(image: str) -> str:
    slug = image.split("/")[-1].split(":")[0]
    if slug in ("busybox", "python", "postgres"):
        return slug
    if slug.startswith("python"):
        return "python"
    if slug.startswith("postgres"):
        return "postgres"
    return slug
