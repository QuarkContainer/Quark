"""Suite execution engine."""

from __future__ import annotations

from keska_lab.backends.base import SandboxBackend
from keska_lab.harness.case import PAUSE_GROUP_CASES, BenchCase
from keska_lab.harness.context import CaseContext
from keska_lab.harness.report import SuiteReport
from keska_lab.harness.stats import MetricStats
from keska_lab.harness.suites import resolve_suite
from keska_lab.harness.workload import WorkloadSpec, get_workload
from keska_lab.setup.quark_cleanup import OrphanReport, cleanup_quark_sandboxes, scan_orphans


def _record_orphans(
    report: SuiteReport,
    orphans: OrphanReport,
    *,
    when: str,
    verbose: bool,
) -> None:
    if orphans.is_empty():
        return
    report.warnings.extend(orphans.warning_lines(when=when))
    if verbose:
        for line in report.warnings[-len(orphans.warning_lines(when=when)) :]:
            print(f"  WARNING: {line}")


def _check_orphans(
    backend: SandboxBackend,
    report: SuiteReport,
    *,
    when: str,
    verbose: bool,
) -> None:
    orphans = scan_orphans(backend.remote)
    _record_orphans(report, orphans, when=when, verbose=verbose)


IO_FS_GROUP = frozenset({"io_write_mib_s", "io_read_mib_s"})


def _backend_supports(backend: SandboxBackend, case: BenchCase) -> bool:
    if "tsot" in case.requires or "network" in case.requires:
        probe = backend.probe()
        if not probe.get("tsot_ready") and not probe.get("network_ready"):
            return False
    if case.name.startswith("io_"):
        if case.name == "io_concurrent_read_mib_s":
            return callable(getattr(backend, "io_fs_concurrent_read_once", None))
        return callable(getattr(backend, "io_fs_once", None))
    if "nsenter" in case.requires:
        if not backend.remote.which("nsenter"):
            return False
        return callable(getattr(backend, "exec_nsenter_batch", None))
    if case.name == "vm_boot_ms":
        return callable(getattr(backend, "vm_boot_batch", None))
    if case.name == "tti_under_load_ms":
        return callable(getattr(backend, "tti_under_load_once", None))
    if case.name in PAUSE_GROUP_CASES:
        return callable(getattr(backend, "pause_resume_once", None))
    if case.name.startswith("memory_idle"):
        return callable(getattr(backend, "memory_idle_once", None))
    if case.name == "cpu_loop_ms":
        return callable(getattr(backend, "cpu_loop_once", None))
    if case.name == "exec_in_running_ms":
        return callable(getattr(backend, "exec_hot_once", None))
    if case.name in {"getpid_ns", "mmap_anon_fault_ms", "pipe_throughput_mib_s"}:
        return callable(getattr(backend, "micro_bench_once", None))
    if case.name == "pgbench_tps":
        return callable(getattr(backend, "pgbench_tps_once", None))
    if case.name.startswith("inet_") or case.name.startswith("sandbox_iperf"):
        return callable(getattr(backend, "inet_tcp_connect_once_ms", None))
    return True


def _batch_fn_name(case: BenchCase) -> str | None:
    mapping = {
        "vm_boot_ms": "vm_boot_batch",
        "tti_ms": "tti_batch",
        "tti_under_load_ms": "tti_under_load_batch",
        "memory_idle_rss_mb": "memory_idle_batch",
        "cpu_loop_ms": "cpu_loop_batch",
        "exec_in_running_ms": "exec_hot_batch",
        "exec_nsenter_ms": "exec_nsenter_batch",
        "getpid_ns": "micro_bench_batch",
        "mmap_anon_fault_ms": "micro_bench_batch",
        "pipe_throughput_mib_s": "micro_bench_batch",
    }
    return mapping.get(case.name)


def _run_case_batch(ctx: CaseContext, case: BenchCase, n: int) -> list[float]:
    fn_name = _batch_fn_name(case)
    if not fn_name:
        raise NotImplementedError(case.name)
    fn = getattr(ctx.backend, fn_name, None)
    if not callable(fn):
        raise NotImplementedError(fn_name)
    w = ctx.workload
    kwargs: dict = {"n": n, "image": w.image}
    if case.name == "vm_boot_ms":
        kwargs["idle_cmd"] = w.idle_cmd
    elif case.name == "tti_ms":
        kwargs["exec_cmd"] = w.tti_exec
        kwargs["timeout"] = w.tti_timeout
    elif case.name == "tti_under_load_ms":
        kwargs["exec_cmd"] = w.tti_exec
    elif case.name == "memory_idle_rss_mb":
        kwargs["idle_cmd"] = w.idle_cmd
    elif case.name == "cpu_loop_ms":
        kwargs["exec_cmd"] = w.cpu_loop_cmd
    elif case.name in {"exec_in_running_ms", "exec_nsenter_ms"}:
        kwargs["exec_cmd"] = w.tti_exec
        if case.name == "exec_nsenter_ms":
            kwargs["image"] = get_workload("busybox").image
    elif case.name in {"getpid_ns", "mmap_anon_fault_ms", "pipe_throughput_mib_s"}:
        kwargs["metric"] = case.name
        kwargs["image"] = get_workload("python").image
    return fn(**kwargs)


def _run_pause_group_batch(
    ctx: CaseContext, n: int
) -> dict[str, list[float]]:
    fn = getattr(ctx.backend, "pause_resume_batch", None)
    if not callable(fn):
        raise NotImplementedError("pause_resume_batch")
    w = ctx.workload
    triples = fn(n, image=w.image, idle_cmd=w.idle_cmd)
    return {
        "pause_ms": [t[0] for t in triples],
        "resume_ms": [t[1] for t in triples],
        "memory_while_paused_rss_mb": [t[2] for t in triples],
    }


def _run_io_fs_batch(ctx: CaseContext, n: int) -> dict[str, list[float]]:
    fn = getattr(ctx.backend, "io_fs_batch", None)
    if not callable(fn):
        raise NotImplementedError("io_fs_batch")
    writes, reads = fn(n, image=ctx.workload.image)
    return {"io_write_mib_s": writes, "io_read_mib_s": reads}


def _append_notes(report: SuiteReport, backend: SandboxBackend) -> None:
    report.notes.append(
        "pause/resume uses runtime pause/resume (OCI freeze), not Quark memory swap hibernate"
    )
    report.notes.append(
        "exec_in_running_ms measures exec via each runtime's management stack "
        "(quark exec vs ctr task exec); compare exec_nsenter_ms on Kata for a thinner path"
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


def _print_samples(
    case: BenchCase, samples: list[float], *, verbose: bool, n: int
) -> None:
    if not verbose:
        return
    for i, val in enumerate(samples):
        print(f"  {case.name} [{i + 1}/{n}] {val:.1f} {case.unit}")


def _run_samples_loop(
    ctx: CaseContext,
    case: BenchCase,
    n: int,
    *,
    verbose: bool,
    backend: SandboxBackend,
    report: SuiteReport,
) -> list[float]:
    samples: list[float] = []
    for i in range(n):
        ctx._pause_resume_cache = None
        try:
            val = case.run_once(ctx)
            samples.append(val)
            if verbose:
                print(f"  {case.name} [{i + 1}/{n}] {val:.1f} {case.unit}")
        except Exception as e:
            report.errors.append(f"{case.name}: {e}")
            if verbose:
                print(f"  {case.name} [{i + 1}/{n}] error: {e}")
            backend.cleanup()
        finally:
            if case.name == "tti_under_load_ms":
                backend.cleanup()
    return samples


def run_suite(
    backend: SandboxBackend,
    *,
    suite: str,
    workload: str | WorkloadSpec = "busybox",
    n: int = 10,
    verbose: bool = True,
) -> SuiteReport:
    spec = workload if isinstance(workload, WorkloadSpec) else get_workload(workload)
    if suite == "micro" and spec.name != "python":
        spec = get_workload("python")
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

    orphans = cleanup_quark_sandboxes(backend.remote)
    _record_orphans(report, orphans, when="before suite", verbose=verbose)

    if verbose:
        print(f"Suite {suite!r} ({spec.name}) × {n} on {backend.name}")

    light_batch_fn = getattr(backend, "run_light_suite_batch", None)
    if (
        suite == "light"
        and backend.name == "quark"
        and callable(light_batch_fn)
    ):
        try:
            grouped = light_batch_fn(
                n,
                image=spec.image,
                exec_cmd=spec.tti_exec,
                cpu_loop_cmd=spec.cpu_loop_cmd,
                idle_cmd=spec.idle_cmd,
            )
            for case in cases:
                if not _backend_supports(backend, case):
                    report.skipped.append(case.name)
                    if verbose:
                        print(f"  skip {case.name} (not supported on {backend.name})")
                    continue
                samples = grouped.get(case.name, [])
                report.metrics[case.name] = MetricStats.from_values(
                    samples, case.unit
                )
                _print_samples(case, samples, verbose=verbose, n=n)
        except Exception as e:
            report.errors.append(f"light suite batch: {e}")
            if verbose:
                print(f"  light suite batch error: {e}")
            backend.cleanup()
        else:
            backend.cleanup()
            orphans = scan_orphans(backend.remote)
            _record_orphans(report, orphans, when="after suite", verbose=verbose)
        return report

    idx = 0
    while idx < len(cases):
        case = cases[idx]

        if not _backend_supports(backend, case):
            report.skipped.append(case.name)
            if verbose:
                print(f"  skip {case.name} (not supported on {backend.name})")
            idx += 1
            continue

        if case.name == "pause_ms":
            pause_cases = [c for c in cases if c.name in PAUSE_GROUP_CASES]
            batch_fn = getattr(backend, "pause_resume_batch", None)
            grouped: dict[str, list[float]] = {
                c.name: [] for c in pause_cases
            }
            try:
                if callable(batch_fn):
                    grouped = _run_pause_group_batch(ctx, n)
                else:
                    for i in range(n):
                        ctx._pause_resume_cache = None
                        p, r, m = backend.pause_resume_once(
                            image=spec.image, idle_cmd=spec.idle_cmd
                        )
                        grouped["pause_ms"].append(p)
                        grouped["resume_ms"].append(r)
                        grouped["memory_while_paused_rss_mb"].append(m)
                        if verbose:
                            print(
                                f"  pause/resume [{i + 1}/{n}] "
                                f"pause={p:.1f}ms resume={r:.1f}ms rss={m:.1f}MB"
                            )
                for metric_name, samples in grouped.items():
                    matched = next(c for c in pause_cases if c.name == metric_name)
                    report.metrics[metric_name] = MetricStats.from_values(
                        samples, matched.unit
                    )
                    if callable(batch_fn):
                        _print_samples(matched, samples, verbose=verbose, n=n)
            except Exception as e:
                report.errors.append(f"pause/resume group: {e}")
                if verbose:
                    print(f"  pause/resume group error: {e}")
                backend.cleanup()
            idx += 1
            while idx < len(cases) and cases[idx].name in PAUSE_GROUP_CASES:
                idx += 1
            continue

        if case.name == "io_write_mib_s" and callable(
            getattr(backend, "io_fs_batch", None)
        ):
            try:
                grouped = _run_io_fs_batch(ctx, n)
                for metric_name, samples in grouped.items():
                    matched = next(c for c in cases if c.name == metric_name)
                    report.metrics[metric_name] = MetricStats.from_values(
                        samples, matched.unit
                    )
                    _print_samples(matched, samples, verbose=verbose, n=n)
            except Exception as e:
                report.errors.append(f"io_fs group: {e}")
                if verbose:
                    print(f"  io_fs group error: {e}")
                backend.cleanup()
            idx += 1
            while idx < len(cases) and cases[idx].name in IO_FS_GROUP:
                idx += 1
            continue

        samples: list[float] = []
        fn_name = _batch_fn_name(case)
        if fn_name and callable(getattr(backend, fn_name, None)):
            try:
                samples = _run_case_batch(ctx, case, n)
                _print_samples(case, samples, verbose=verbose, n=n)
            except Exception as e:
                report.errors.append(f"{case.name}: {e}")
                if verbose:
                    print(f"  {case.name} batch error: {e}")
                backend.cleanup()
        else:
            samples = _run_samples_loop(
                ctx, case, n, verbose=verbose, backend=backend, report=report
            )

        report.metrics[case.name] = MetricStats.from_values(samples, case.unit)
        _check_orphans(backend, report, when=f"after {case.name}", verbose=verbose)
        idx += 1

    orphans = cleanup_quark_sandboxes(backend.remote)
    _record_orphans(report, orphans, when="after suite", verbose=verbose)
    return report
