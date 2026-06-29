"""Group 5 experimental I/O A/B runner and decision-packet helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from keska_lab.config import LabConfig
from keska_lab.harness.report import SuiteReport
from keska_lab.harness.stats import MetricStats
from keska_lab.provision import ProvisionContext, build_quark, install_quark, sync_sources
from keska_lab.remote import RemoteHost
from keska_lab.session import LabSession
from keska_lab.setup.quark_config import (
    GROUP5_BENCH_MATRIX,
    GROUP5_ITERATIONS,
    QuarkBenchConfigStep,
    QuarkExperimentalConfigStep,
    experimental_cargo_features,
)

# Global veto thresholds (experimental vs baseline).
VETO_GATES: dict[str, dict[str, float | str]] = {
    "tti_ms": {"suite": "light", "p50_pct": 5, "p99_pct": 10},
    "tti_under_load_ms": {"suite": "light", "p50_pct": 5},
    "memory_idle_rss_mb": {"suite": "light,db", "pct": 3, "abs_mb": 8},
    "memory_while_paused_rss_mb": {"suite": "light", "pct": 3, "abs_mb": 8},
    "pause_ms": {"suite": "light", "p50_pct": 10},
    "resume_ms": {"suite": "light", "p50_pct": 10},
}


E2_IMPLEMENT_BARS = {
    "io_read_mib_s": {"suite": "full", "p50_gain_pct": 8},
    "pgbench_tps": {"suite": "db", "p50_gain_pct": 5},
}


@dataclass
class GateResult:
    metric: str
    suite: str
    baseline_p50: float
    experimental_p50: float
    threshold: str
    passed: bool
    note: str = ""


@dataclass
class AbCompareResult:
    flag: str
    baseline: dict[str, SuiteReport]
    experimental: dict[str, SuiteReport]
    veto: list[GateResult] = field(default_factory=list)
    implement: list[GateResult] = field(default_factory=list)

    @property
    def veto_passed(self) -> bool:
        return all(g.passed for g in self.veto)

    @property
    def implement_passed(self) -> bool:
        return all(g.passed for g in self.implement)


def _metric_p50(report: SuiteReport, name: str) -> float | None:
    val = report.metrics.get(name)
    if isinstance(val, MetricStats):
        return val.p50
    return None


def _pct_delta(baseline: float, experimental: float) -> float:
    if baseline == 0:
        return float("inf") if experimental > 0 else 0.0
    return (experimental - baseline) / baseline * 100.0


def _memory_veto(baseline: float, experimental: float, pct: float, abs_mb: float) -> tuple[bool, str]:
    delta = experimental - baseline
    pct_delta = _pct_delta(baseline, experimental)
    ok = delta <= abs_mb and pct_delta <= pct
    thresh = f"≤ baseline +{pct}% and +{abs_mb} MB"
    return ok, thresh


def _p50_veto(baseline: float, experimental: float, pct: float) -> tuple[bool, str]:
    pct_delta = _pct_delta(baseline, experimental)
    ok = pct_delta <= pct
    return ok, f"≤ baseline +{pct}%"


def evaluate_veto(baseline: dict[str, SuiteReport], experimental: dict[str, SuiteReport]) -> list[GateResult]:
    results: list[GateResult] = []
    for metric, spec in VETO_GATES.items():
        suites = [s.strip() for s in str(spec["suite"]).split(",")]
        for suite in suites:
            b_rep = baseline.get(suite)
            e_rep = experimental.get(suite)
            if not b_rep or not e_rep:
                continue
            b = _metric_p50(b_rep, metric)
            e = _metric_p50(e_rep, metric)
            if b is None or e is None:
                continue
            if e_rep.errors or e == 0:
                results.append(
                    GateResult(
                        metric=metric,
                        suite=suite,
                        baseline_p50=b,
                        experimental_p50=e,
                        threshold="valid experimental samples required",
                        passed=False,
                        note=f"errors={len(e_rep.errors)}",
                    )
                )
                continue
            if "abs_mb" in spec:
                ok, thresh = _memory_veto(b, e, float(spec["pct"]), float(spec["abs_mb"]))
            else:
                ok, thresh = _p50_veto(b, e, float(spec.get("p50_pct", spec.get("pct", 0))))
            results.append(
                GateResult(
                    metric=metric,
                    suite=suite,
                    baseline_p50=b,
                    experimental_p50=e,
                    threshold=thresh,
                    passed=ok,
                    note=f"delta {_pct_delta(b, e):+.1f}%",
                )
            )
    return results


def evaluate_e2_implement(baseline: dict[str, SuiteReport], experimental: dict[str, SuiteReport]) -> list[GateResult]:
    results: list[GateResult] = []
    gains: list[bool] = []

    full_b = baseline.get("full")
    full_e = experimental.get("full")
    if full_b and full_e:
        spec = E2_IMPLEMENT_BARS["io_read_mib_s"]
        b = _metric_p50(full_b, "io_read_mib_s")
        e = _metric_p50(full_e, "io_read_mib_s")
        if b is not None and e is not None and b != 0:
            gain = (e - b) / b * 100.0
            ok = gain >= float(spec["p50_gain_pct"])
            gains.append(ok)
            results.append(
                GateResult(
                    metric="io_read_mib_s",
                    suite="full",
                    baseline_p50=b,
                    experimental_p50=e,
                    threshold=f"≥ +{spec['p50_gain_pct']}% p50",
                    passed=ok,
                    note=f"gain {gain:+.1f}%",
                )
            )

    db_b = baseline.get("db")
    db_e = experimental.get("db")
    if db_b and db_e:
        spec = E2_IMPLEMENT_BARS["pgbench_tps"]
        b = _metric_p50(db_b, "pgbench_tps")
        e = _metric_p50(db_e, "pgbench_tps")
        if b is not None and e is not None and b != 0:
            gain = (e - b) / b * 100.0
            ok = gain >= float(spec["p50_gain_pct"])
            gains.append(ok)
            results.append(
                GateResult(
                    metric="pgbench_tps",
                    suite="db",
                    baseline_p50=b,
                    experimental_p50=e,
                    threshold=f"≥ +{spec['p50_gain_pct']}% p50",
                    passed=ok,
                    note=f"gain {gain:+.1f}%",
                )
            )

    if gains and not any(gains):
        results.append(
            GateResult(
                metric="io_read_or_pgbench",
                suite="full,db",
                baseline_p50=0,
                experimental_p50=0,
                threshold="≥8% io_read or ≥5% pgbench_tps",
                passed=False,
                note="neither metric met bar",
            )
        )
    return results


def compare_ab(
    flag: str,
    baseline: dict[str, SuiteReport],
    experimental: dict[str, SuiteReport],
) -> AbCompareResult:
    result = AbCompareResult(flag=flag, baseline=baseline, experimental=experimental)
    result.veto = evaluate_veto(baseline, experimental)
    if flag == "MmapRead":
        result.implement = evaluate_e2_implement(baseline, experimental)
    return result


def print_ab_compare(result: AbCompareResult) -> None:
    from keska_lab.display import console

    console.print(f"\n[bold]Group 5 A/B: {result.flag}[/bold]")
    for suite in sorted(set(result.baseline) | set(result.experimental)):
        b = result.baseline.get(suite)
        e = result.experimental.get(suite)
        if not b or not e:
            continue
        console.print(f"\n[cyan]{suite}[/cyan]")
        for name in b.metrics:
            bv = _metric_p50(b, name)
            ev = _metric_p50(e, name)
            if bv is None or ev is None:
                continue
            ratio = ev / bv if bv else math.inf
            console.print(f"  {name}: baseline {bv} → experimental {ev} ({ratio:.3f}x)")

    console.print("\n[bold]Veto gates[/bold]")
    for g in result.veto:
        status = "[green]PASS[/green]" if g.passed else "[red]FAIL[/red]"
        console.print(
            f"  {status} {g.metric} ({g.suite}): {g.baseline_p50} → {g.experimental_p50} ({g.note})"
        )

    if result.implement:
        console.print("\n[bold]Implement bars[/bold]")
        for g in result.implement:
            status = "[green]PASS[/green]" if g.passed else "[red]FAIL[/red]"
            console.print(
                f"  {status} {g.metric} ({g.suite}): {g.baseline_p50} → {g.experimental_p50} ({g.note})"
            )


def _save_reports(reports: dict[str, SuiteReport], tag: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    out_dir = Path.home() / ".keska-lab" / "results" / "group5"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    for suite, rep in reports.items():
        path = out_dir / f"{ts}_{tag}_{rep.backend}_{suite}.json"
        rep.save(path)
        paths[suite] = str(path)
    return paths


def rebuild_quark(
    remote: RemoteHost,
    config: LabConfig,
    *,
    cargo_features: str = "",
    stream: bool = True,
) -> None:
    ctx = ProvisionContext(
        remote=remote,
        config=config,
        repo=config.resolve_local_repo(),
        profile=config.quark_build_profile,
        stream=stream,
    )
    sync_sources(ctx)
    build_quark(ctx, cargo_features=cargo_features)
    install_quark(ctx)


def prepare_group5(lab: LabSession) -> None:
    """One-time lab prep for light/full/db suites (bundles, io dir, postgres template)."""
    lab.quark.prepare(stream=False, mode="full", workload="busybox")
    lab.quark.prepare(stream=False, mode="db")


def _load_suite_report(path: Path) -> SuiteReport:
    data = json.loads(path.read_text())
    rep = SuiteReport(
        backend=data["backend"],
        host=data["host"],
        timestamp=data["timestamp"],
        suite=data["suite"],
        workload=data["workload"],
        iterations=data["iterations"],
        image=data["image"],
        errors=data.get("errors", []),
        notes=data.get("notes", []),
        skipped=data.get("skipped", []),
    )
    for name, val in data.get("metrics", {}).items():
        if isinstance(val, dict):
            rep.metrics[name] = MetricStats(**val)
        else:
            rep.metrics[name] = val
    return rep


def load_baseline_from_dir(flag: str, directory: Path | None = None) -> tuple[dict[str, SuiteReport], dict[str, str]]:
    """Load most recent baseline JSON artifacts for a flag."""
    root = directory or (Path.home() / ".keska-lab" / "results" / "group5")
    pattern = f"*_baseline_{flag}_quark_*.json"
    paths = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"no baseline artifacts matching {pattern} under {root}")
    by_suite: dict[str, SuiteReport] = {}
    artifact_paths: dict[str, str] = {}
    seen: set[str] = set()
    for path in paths:
        suite = path.stem.rsplit("_", 1)[-1]
        if suite in seen:
            continue
        seen.add(suite)
        by_suite[suite] = _load_suite_report(path)
        artifact_paths[suite] = str(path)
    return by_suite, artifact_paths


def run_quark_suites(
    lab: LabSession,
    suites: tuple[str, ...],
    *,
    n: int = GROUP5_ITERATIONS,
    setup: bool = False,
) -> dict[str, SuiteReport]:
    reports: dict[str, SuiteReport] = {}
    for suite in suites:
        lab.backend.cleanup()
        rep = lab.quark.bench(suite, n=n, setup=setup, save=True)
        reports[suite] = rep
    return reports


def run_group5_experimental_arm(
    flag: str,
    *,
    n: int = GROUP5_ITERATIONS,
    stream: bool = True,
) -> dict[str, SuiteReport]:
    """Build experimental binary, deploy config, run suites (baseline must already exist)."""
    if flag not in GROUP5_BENCH_MATRIX:
        raise ValueError(f"unknown flag {flag!r}")
    suites = GROUP5_BENCH_MATRIX[flag]
    lab = LabSession()
    features = ",".join(experimental_cargo_features(flag))
    rebuild_quark(lab.remote, lab.config, cargo_features=features, stream=stream)
    QuarkExperimentalConfigStep(flag).run(lab.remote, stream=stream)
    return run_quark_suites(lab, suites, n=n, setup=False)


def run_group5_ab(
    flag: str,
    *,
    n: int = GROUP5_ITERATIONS,
    skip_baseline_build: bool = False,
    experimental_only: bool = False,
    baseline_reports: dict[str, SuiteReport] | None = None,
    stream: bool = True,
) -> AbCompareResult:
    """Run baseline then experimental Quark benches for one Group 5 flag."""
    if flag not in GROUP5_BENCH_MATRIX:
        raise ValueError(f"unknown flag {flag!r}")
    suites = GROUP5_BENCH_MATRIX[flag]
    lab = LabSession()
    remote = lab.remote
    config = lab.config
    features = ",".join(experimental_cargo_features(flag))

    prepare_group5(lab)

    baseline_paths: dict[str, str]
    if experimental_only:
        if baseline_reports is not None:
            baseline = baseline_reports
            baseline_paths = {}
        else:
            baseline, baseline_paths = load_baseline_from_dir(flag)
    else:
        if not skip_baseline_build:
            rebuild_quark(remote, config, cargo_features="", stream=stream)
        QuarkBenchConfigStep().run(remote, stream=stream)
        baseline = run_quark_suites(lab, suites, n=n, setup=False)
        baseline_paths = _save_reports(baseline, f"baseline_{flag}")

    if experimental_only:
        experimental = run_group5_experimental_arm(flag, n=n, stream=stream)
    else:
        rebuild_quark(remote, config, cargo_features=features, stream=stream)
        QuarkExperimentalConfigStep(flag).run(remote, stream=stream)
        experimental = run_quark_suites(lab, suites, n=n, setup=False)
    experimental_paths = _save_reports(experimental, f"experimental_{flag}")

    result = compare_ab(flag, baseline, experimental)
    print_ab_compare(result)

    packet_dir = Path.home() / ".keska-lab" / "results" / "group5"
    packet_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    packet_path = packet_dir / f"{ts}_{flag}_decision_packet.json"
    packet: dict[str, Any] = {
        "flag": flag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iterations": n,
        "suites": list(suites),
        "cargo_features": features,
        "baseline_artifacts": baseline_paths,
        "experimental_artifacts": experimental_paths,
        "veto_passed": result.veto_passed,
        "implement_passed": result.implement_passed,
        "veto": [g.__dict__ for g in result.veto],
        "implement": [g.__dict__ for g in result.implement],
        "recommendation": _recommendation(result),
    }
    packet_path.write_text(json.dumps(packet, indent=2) + "\n")
    from keska_lab.display import console

    console.print(f"\n[dim]Decision packet: {packet_path}[/dim]")
    console.print(f"[bold]Recommendation:[/bold] {packet['recommendation']}")
    console.print("[dim]No code changes until you explicitly approve promote or delete.[/dim]")
    return result


def _recommendation(result: AbCompareResult) -> str:
    if result.veto_passed and result.implement_passed:
        return "promote (pending your explicit approval)"
    if not result.veto_passed:
        return "delete (veto failed — pending your explicit approval)"
    return "delete or defer (implement bars not met — pending your explicit approval)"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Group 5 experimental I/O A/B on lab host")
    parser.add_argument(
        "flag",
        choices=list(GROUP5_BENCH_MATRIX),
        help="experimental flag to test",
    )
    parser.add_argument("-n", type=int, default=GROUP5_ITERATIONS, help="iterations per suite")
    parser.add_argument(
        "--skip-baseline-build",
        action="store_true",
        help="assume baseline binary already installed",
    )
    parser.add_argument(
        "--experimental-only",
        action="store_true",
        help="skip baseline run; load latest baseline JSON from ~/.keska-lab/results/group5",
    )
    args = parser.parse_args()
    try:
        run_group5_ab(
            args.flag,
            n=args.n,
            skip_baseline_build=args.skip_baseline_build,
            experimental_only=args.experimental_only,
            stream=True,
        )
    except Exception as exc:
        print(f"group5 A/B failed: {exc}", file=__import__("sys").stderr)
        return 1
    return 0
