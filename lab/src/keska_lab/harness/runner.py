"""Suite execution engine."""

from __future__ import annotations

from keska_lab.backends.base import SandboxBackend
from keska_lab.harness.case import BenchCase
from keska_lab.harness.context import CaseContext
from keska_lab.harness.report import SuiteReport
from keska_lab.harness.stats import MetricStats
from keska_lab.harness.suites import resolve_suite
from keska_lab.harness.workload import WorkloadSpec, get_workload


def _backend_supports(backend: SandboxBackend, case: BenchCase) -> bool:
    if "tsot" in case.requires or "network" in case.requires:
        probe = backend.probe()
        if not probe.get("tsot_ready") and not probe.get("network_ready"):
            return False
    if case.name.startswith("io_"):
        if case.name == "io_concurrent_read_mib_s":
            return callable(getattr(backend, "io_fs_concurrent_read_once", None))
        return callable(getattr(backend, "io_fs_once", None))
    if case.name == "tti_under_load_ms":
        return callable(getattr(backend, "tti_under_load_once", None))
    if case.name.startswith("pause") or case.name.startswith("resume") or "paused" in case.name:
        return callable(getattr(backend, "pause_resume_once", None))
    if case.name.startswith("memory_idle"):
        return callable(getattr(backend, "memory_idle_once", None))
    if case.name == "pgbench_tps":
        return callable(getattr(backend, "pgbench_tps_once", None))
    if case.name.startswith("inet_") or case.name.startswith("sandbox_iperf"):
        return callable(getattr(backend, "inet_tcp_connect_once_ms", None))
    return True


def _append_notes(report: SuiteReport, backend: SandboxBackend) -> None:
    report.notes.append(
        "pause/resume uses runtime pause/resume (OCI freeze), not Quark memory swap hibernate"
    )
    if backend.name == "quark":
        probe = backend.probe()
        if probe.get("tsot_ready"):
            report.notes.append(
                "Quark network via CRI/crictl (TSOT); direct OCI for non-network suites"
            )
        else:
            report.notes.append(
                f"Quark direct OCI via {getattr(backend, 'quark_bin', 'quark')} (no docker in hot path)"
            )
    if backend.name == "kata":
        cfg = getattr(backend, "config", None)
        hv = getattr(cfg, "kata_hypervisor", "firecracker")
        snap = getattr(cfg, "kata_snapshotter", None)
        extra = f", snapshotter={snap}" if snap else ""
        probe = backend.probe()
        if probe.get("network_ready"):
            report.notes.append(f"Kata ({hv}) network via CRI/crictl (runtime_handler=kata)")
        else:
            report.notes.append(f"Kata ({hv}) via containerd ctr{extra} (no dockerd in hot path)")


def run_suite(
    backend: SandboxBackend,
    *,
    suite: str,
    workload: str | WorkloadSpec = "busybox",
    n: int = 10,
    verbose: bool = True,
) -> SuiteReport:
    spec = workload if isinstance(workload, WorkloadSpec) else get_workload(workload)
    cases = resolve_suite(suite)
    remote = backend.remote
    report = SuiteReport.new(
        backend=backend.name,
        host=remote.config.ssh_target,
        suite=suite,
        workload=spec.name,
        image=spec.image,
        iterations=n,
    )
    _append_notes(report, backend)
    ctx = CaseContext(backend=backend, workload=spec, verbose=verbose)

    backend.cleanup()

    if verbose:
        print(f"Suite {suite!r} ({spec.name}) × {n} on {backend.name}")

    for case in cases:
        if not _backend_supports(backend, case):
            report.skipped.append(case.name)
            if verbose:
                print(f"  skip {case.name} (not supported on {backend.name})")
            continue

        samples: list[float] = []
        for i in range(n):
            try:
                val = case.run_once(ctx)
                samples.append(val)
                if verbose:
                    print(f"  {case.name} [{i + 1}/{n}] {val:.1f} {case.unit}")
            except Exception as e:
                report.errors.append(f"{case.name}: {e}")
                if verbose:
                    print(f"  {case.name} [{i + 1}/{n}] error: {e}")
        report.metrics[case.name] = MetricStats.from_values(samples, case.unit)

    backend.cleanup()
    return report
