from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import psutil


@dataclass(frozen=True)
class ProcMatch:
    pid: int
    name: str
    cmdline: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OrphanReport:
    shims: list[ProcMatch] = field(default_factory=list)
    firecrackers: list[ProcMatch] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.shims and not self.firecrackers


def _safe_cmdline(p: psutil.Process) -> list[str]:
    try:
        return list(p.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return []


def scan_orphans() -> OrphanReport:
    shims: list[ProcMatch] = []
    firecrackers: list[ProcMatch] = []
    for p in psutil.process_iter(attrs=["pid", "name"]):
        try:
            name = (p.info.get("name") or "").strip()
            if name == "firecracker":
                firecrackers.append(
                    ProcMatch(pid=p.pid, name=name, cmdline=_safe_cmdline(p))
                )
                continue
            if name.startswith("containerd-shim-quark"):
                shims.append(ProcMatch(pid=p.pid, name=name, cmdline=_safe_cmdline(p)))
                continue
            # Some distros report shim name generically; fall back to cmdline match.
            cmd = _safe_cmdline(p)
            if cmd and any("containerd-shim-quark" in part for part in cmd[:2]):
                shims.append(ProcMatch(pid=p.pid, name=name or "containerd-shim-quark", cmdline=cmd))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return OrphanReport(shims=shims, firecrackers=firecrackers)


def terminate_processes(procs: Iterable[ProcMatch], *, timeout_s: float = 3.0) -> None:
    ps: list[psutil.Process] = []
    for m in procs:
        try:
            ps.append(psutil.Process(m.pid))
        except psutil.NoSuchProcess:
            continue

    for p in ps:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    gone, alive = psutil.wait_procs(ps, timeout=timeout_s)
    _ = gone
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def drain_orphans(*, budget_s: float = 5.0) -> OrphanReport:
    """Best-effort drain of shim/firecracker processes within a small budget.

    This is intentionally bounded. If processes survive, callers should treat it as a leak.
    """
    import time

    deadline = time.monotonic() + budget_s
    rep = scan_orphans()
    while time.monotonic() < deadline:
        if rep.is_empty():
            return rep
        # Try a gentle terminate first; escalate on next loop via terminate_processes kill path.
        terminate_processes([*rep.shims, *rep.firecrackers], timeout_s=0.5)
        time.sleep(0.2)
        rep = scan_orphans()
    return rep

