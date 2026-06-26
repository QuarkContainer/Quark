"""Named benchmark suites."""

from __future__ import annotations

from keska_lab.harness.case import (
    CASE_INET_CONNECT,
    CASE_INET_DOWNLOAD,
    CASE_IO_READ,
    CASE_IO_WRITE,
    CASE_MEM_IDLE,
    CASE_MEM_PAUSED,
    CASE_PAUSE,
    CASE_PGBENCH_TPS,
    CASE_RESUME,
    CASE_SANDBOX_IPERF,
    CASE_TTI,
    CASE_TTI_LOAD,
    BenchCase,
)

LIGHT_CASES: tuple[BenchCase, ...] = (
    CASE_TTI,
    CASE_TTI_LOAD,
    CASE_MEM_IDLE,
    CASE_PAUSE,
    CASE_RESUME,
    CASE_MEM_PAUSED,
)

WORKLOADS_CASES: tuple[BenchCase, ...] = (
    CASE_TTI,
    CASE_MEM_IDLE,
)

NETWORK_CASES: tuple[BenchCase, ...] = (
    CASE_INET_CONNECT,
    CASE_INET_DOWNLOAD,
    CASE_SANDBOX_IPERF,
)

STANDARD_CASES: tuple[BenchCase, ...] = LIGHT_CASES

HEAVY_CASES: tuple[BenchCase, ...] = (
    *LIGHT_CASES,
    CASE_IO_WRITE,
    CASE_IO_READ,
    *NETWORK_CASES,
)

DB_CASES: tuple[BenchCase, ...] = (
    CASE_TTI,
    CASE_MEM_IDLE,
    CASE_PGBENCH_TPS,
)

SUITE_MODES: dict[str, tuple[BenchCase, ...]] = {
    "light": LIGHT_CASES,
    "full": LIGHT_CASES,
    "workloads": WORKLOADS_CASES,
    "network": NETWORK_CASES,
    "standard": STANDARD_CASES,
    "heavy": HEAVY_CASES,
    "db": DB_CASES,
}


def resolve_suite(mode: str) -> tuple[BenchCase, ...]:
    key = mode.lower().strip()
    if key not in SUITE_MODES:
        known = ", ".join(sorted(SUITE_MODES))
        raise ValueError(f"unknown suite mode {mode!r}; choose from: {known}")
    return SUITE_MODES[key]
