"""Benchmark case definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from keska_lab.harness.context import CaseContext
from keska_lab.harness.workload import get_workload


@dataclass(frozen=True)
class BenchCase:
    name: str
    unit: str
    requires: frozenset[str] = frozenset()
    run: Callable[[CaseContext], float] | None = None

    def run_once(self, ctx: CaseContext) -> float:
        if self.run is None:
            raise NotImplementedError(self.name)
        return self.run(ctx)


def _tti(ctx: CaseContext) -> float:
    w = ctx.workload
    return ctx.backend.tti_once(image=w.image, exec_cmd=w.tti_exec, timeout=w.tti_timeout)


def _tti_under_load(ctx: CaseContext) -> float:
    w = ctx.workload
    return ctx.backend.tti_under_load_once(image=w.image, exec_cmd=w.tti_exec)


def _memory_idle(ctx: CaseContext) -> float:
    w = ctx.workload
    return ctx.backend.memory_idle_once(image=w.image, idle_cmd=w.idle_cmd)


def _pause_ms(ctx: CaseContext) -> float:
    w = ctx.workload
    p, _, _ = ctx.backend.pause_resume_once(image=w.image, idle_cmd=w.idle_cmd)
    return p


def _resume_ms(ctx: CaseContext) -> float:
    w = ctx.workload
    _, r, _ = ctx.backend.pause_resume_once(image=w.image, idle_cmd=w.idle_cmd)
    return r


def _memory_paused(ctx: CaseContext) -> float:
    w = ctx.workload
    _, _, m = ctx.backend.pause_resume_once(image=w.image, idle_cmd=w.idle_cmd)
    return m


def _io_write(ctx: CaseContext) -> float:
    write_mib_s, _ = ctx.backend.io_fs_once(image=ctx.workload.image)
    return write_mib_s


def _io_read(ctx: CaseContext) -> float:
    _, read_mib_s = ctx.backend.io_fs_once(image=ctx.workload.image)
    return read_mib_s


def _io_concurrent_read(ctx: CaseContext) -> float:
    fn = getattr(ctx.backend, "io_fs_concurrent_read_once", None)
    if not callable(fn):
        raise NotImplementedError("io_fs_concurrent_read_once")
    return fn(image=ctx.workload.image)


def _inet_connect(ctx: CaseContext) -> float:
    return ctx.backend.inet_tcp_connect_once_ms(image=ctx.workload.image)


def _inet_download(ctx: CaseContext) -> float:
    return ctx.backend.inet_download_mbps_once(image=ctx.workload.image)


def _sandbox_iperf(ctx: CaseContext) -> float:
    return ctx.backend.sandbox_iperf_mbps_once(image=get_workload("iperf").image)


def _pgbench_tps(ctx: CaseContext) -> float:
    w = ctx.workload
    return ctx.backend.pgbench_tps_once(image=w.image, idle_cmd=w.idle_cmd)


CASE_TTI = BenchCase("tti_ms", "ms", run=_tti)
CASE_TTI_LOAD = BenchCase("tti_under_load_ms", "ms", run=_tti_under_load)
CASE_MEM_IDLE = BenchCase("memory_idle_rss_mb", "MB", run=_memory_idle)
CASE_PAUSE = BenchCase("pause_ms", "ms", run=_pause_ms)
CASE_RESUME = BenchCase("resume_ms", "ms", run=_resume_ms)
CASE_MEM_PAUSED = BenchCase("memory_while_paused_rss_mb", "MB", run=_memory_paused)
CASE_IO_WRITE = BenchCase("io_write_mib_s", "MiB/s", run=_io_write)
CASE_IO_READ = BenchCase("io_read_mib_s", "MiB/s", run=_io_read)
CASE_IO_CONCURRENT_READ = BenchCase(
    "io_concurrent_read_mib_s", "MiB/s", run=_io_concurrent_read
)
CASE_INET_CONNECT = BenchCase("inet_tcp_connect_ms", "ms", frozenset({"tsot", "network"}), _inet_connect)
CASE_INET_DOWNLOAD = BenchCase("inet_download_mbps", "Mbits/s", frozenset({"tsot", "network"}), _inet_download)
CASE_SANDBOX_IPERF = BenchCase(
    "sandbox_iperf_mbps", "Mbits/s", frozenset({"tsot", "network"}), _sandbox_iperf
)
CASE_PGBENCH_TPS = BenchCase("pgbench_tps", "TPS", run=_pgbench_tps)
