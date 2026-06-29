"""Group 5 (E2/E3): experimental I/O lab config and benchmark protocol checks."""

from __future__ import annotations

import pytest

from keska_lab.group5 import VETO_GATES, evaluate_veto
from keska_lab.harness.report import SuiteReport
from keska_lab.harness.stats import MetricStats
from keska_lab.setup.quark_config import (
    EXPERIMENTAL_FLAG_KEYS,
    GROUP5_BENCH_MATRIX,
    GROUP5_ITERATIONS,
    GROUP5_SUITES,
    bench_config_json,
    experimental_cargo_features,
    experimental_config_json,
)


@pytest.mark.parametrize("flag", EXPERIMENTAL_FLAG_KEYS)
def test_experimental_config_enables_single_flag(flag: str) -> None:
    cfg = experimental_config_json(flag)
    prod = bench_config_json()
    for key in EXPERIMENTAL_FLAG_KEYS:
        assert cfg[key] is (key == flag)
    for key, value in prod.items():
        if key not in EXPERIMENTAL_FLAG_KEYS:
            assert cfg[key] == value


@pytest.mark.parametrize("flag", EXPERIMENTAL_FLAG_KEYS)
def test_experimental_cargo_features(flag: str) -> None:
    features = experimental_cargo_features(flag)
    assert features
    assert all(f.startswith("experimental-") for f in features)


def test_experimental_config_rejects_unknown_flag() -> None:
    with pytest.raises(ValueError, match="unknown experimental flag"):
        experimental_config_json("FileBuf")


@pytest.mark.parametrize("flag", EXPERIMENTAL_FLAG_KEYS)
def test_group5_unified_benchmark_matrix(flag: str) -> None:
    assert GROUP5_BENCH_MATRIX[flag] == GROUP5_SUITES
    assert len(GROUP5_SUITES) == 3
    assert GROUP5_ITERATIONS >= 5


def _suite(**metrics: float) -> SuiteReport:
    rep = SuiteReport.new(
        backend="quark",
        host="lab",
        suite="light",
        workload="busybox",
        image="busybox",
        iterations=5,
    )
    for name, p50 in metrics.items():
        rep.metrics[name] = MetricStats.from_values([p50] * 5, "ms" if name.endswith("_ms") else "MB")
    return rep


def test_veto_memory_absolute_cap() -> None:
    baseline = {"light": _suite(memory_idle_rss_mb=100, memory_while_paused_rss_mb=100)}
    experimental = {"light": _suite(memory_idle_rss_mb=109, memory_while_paused_rss_mb=109)}
    gates = evaluate_veto(baseline, experimental)
    mem = [g for g in gates if g.metric == "memory_idle_rss_mb"][0]
    assert mem.passed is False


def test_veto_gates_defined_for_light_metrics() -> None:
    assert "tti_ms" in VETO_GATES
    assert "memory_while_paused_rss_mb" in VETO_GATES
