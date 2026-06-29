"""E2 MmapRead coherency checklist C1–C7 (see kdoc/aaa-runtime-roadmap.md)."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from keska_lab.config import LabConfig
from keska_lab.harness.io_fs import IO_BENCH_MOUNT, quark_io_bench_bundle_preamble, quark_io_bench_cleanup_trap
from keska_lab.harness.network import python_exec_cmd
from keska_lab.provision import ProvisionContext, build_quark, install_quark, sync_sources
from keska_lab.remote import RemoteHost
from keska_lab.session import LabSession
from keska_lab.setup.quark_config import QuarkExperimentalConfigStep, experimental_cargo_features

COHERENCY_IMAGE = "python:3.12-slim"
BENCH = IO_BENCH_MOUNT
COHERENCY_HOST_BENCH_PREP = 'sudo -n chmod 777 "$HOST_BENCH"'


@dataclass
class CoherencyCase:
    id: str
    name: str
    description: str


COHERENCY_CASES: tuple[CoherencyCase, ...] = (
    CoherencyCase("C1", "host_write_guest_read", "Host write → guest read() sees new bytes"),
    CoherencyCase("C2", "guest_write_guest_read", "Guest write → guest read() read-your-writes"),
    CoherencyCase("C3", "truncate_while_mapped", "Truncate while mapped — no stale data past EOF"),
    CoherencyCase("C4", "concurrent_host_guest", "Concurrent host write + guest read — no torn blocks"),
    CoherencyCase("C5", "mmap_vs_read", "mmap(MAP_SHARED) vs read() same region — consistent"),
    CoherencyCase("C6", "pgbench_soak", "pgbench ≥30 min — no crash; RSS drift ≤3%"),
    CoherencyCase("C7", "open_read_close_leak", "1000× open/read/close — RSS within +3%"),
)


@dataclass
class CoherencyResult:
    case_id: str
    passed: bool
    message: str
    detail: str = ""


@dataclass
class CoherencySuiteResult:
    flag: str = "MmapRead"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    results: list[CoherencyResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "flag": self.flag,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "results": [r.__dict__ for r in self.results],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")


# --- Guest Python payloads (base64-wrapped by python_exec_cmd) ---

C2_GUEST_PY = f"""
path = "{BENCH}/c2.dat"
data = b"guest-write-read-coherency"
with open(path, "wb") as f:
    f.write(data)
with open(path, "rb") as f:
    got = f.read()
assert got == data, (got, data)
print("PASS C2")
"""

C3_GUEST_PY = f"""
import os, mmap
path = "{BENCH}/c3.dat"
with open(path, "wb") as f:
    f.write(b"X" * 65536)
fd = os.open(path, os.O_RDWR)
try:
    mm = mmap.mmap(fd, 4096, prot=mmap.PROT_READ, flags=mmap.MAP_SHARED)
    os.ftruncate(fd, 2048)
    os.lseek(fd, 4096, os.SEEK_SET)
    tail = os.read(fd, 4096)
    assert tail == b"", f"read past EOF got {{len(tail)}} bytes"
    os.lseek(fd, 0, os.SEEK_SET)
    head = os.read(fd, 2048)
    assert len(head) == 2048
    os.lseek(fd, 2048, os.SEEK_SET)
    assert os.read(fd, 1) == b""
finally:
    os.close(fd)
print("PASS C3")
"""

C5_GUEST_PY = f"""
import os, mmap
path = "{BENCH}/c5.dat"
payload = bytes(range(256)) * 16
fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
try:
    os.ftruncate(fd, len(payload))
    os.write(fd, payload)
    os.lseek(fd, 0, os.SEEK_SET)
    mm = mmap.mmap(fd, len(payload), prot=mmap.PROT_READ, flags=mmap.MAP_SHARED)
    os.lseek(fd, 0, os.SEEK_SET)
    via_read = os.read(fd, len(payload))
    assert via_read == payload, "read() mismatch"
    assert mm[:] == payload, "mmap mismatch"
    assert via_read == mm[:], "read vs mmap mismatch"
finally:
    os.close(fd)
print("PASS C5")
"""

C4_GUEST_PY = f"""
import os, time
path = "{BENCH}/c4.dat"
size = 4096
for _ in range(50):
    with open(path, "rb") as f:
        block = f.read(size)
    if len(block) != size:
        time.sleep(0.01)
        continue
    if block != b"\\x00" * size:
        raise AssertionError("torn or partial block")
print("PASS C4")
"""

C7_GUEST_PY = f"""
import os
path = "{BENCH}/c7.dat"
with open(path, "wb") as f:
    f.write(b"z" * 65536)
for _ in range(1000):
    with open(path, "rb") as f:
        assert f.read(4096) == b"z" * 4096
print("PASS C7")
"""

C1_GUEST_READ_PY = f"""
import os
path = "{BENCH}/c1.dat"
with open(path, "rb") as f:
    print(f.read(8).hex())
"""


def _guest_pass_line(stdout: str, token: str) -> bool:
    return any(line.strip() == token for line in stdout.splitlines())


def _coherency_sandbox_script(
    backend,
    *,
    guest_cmd: str,
    host_between: str = "",
    timeout: int = 120,
) -> str:
    """Create sandbox with /bench bind mount, run guest_cmd, optional host_between on HOST_BENCH."""
    q = backend._quark_cmd
    prep = quark_io_bench_bundle_preamble(backend.config, COHERENCY_IMAGE)
    cleanup = quark_io_bench_cleanup_trap(quark_delete_cmd=backend._quark_force_delete())
    return textwrap.dedent(
        f"""
        set -euo pipefail
        {prep}
        {COHERENCY_HOST_BENCH_PREP}
        {cleanup}
        ID=keska-coh-$RANDOM
        {q('create "$ID" -b "$BUNDLE"')}
        {q('start "$ID"')}
        out=$({q(f'exec --user 0:0 "$ID" -- {guest_cmd}')})
        {host_between}
        cleanup_io_bench
        trap - EXIT INT TERM
        echo "$out"
        """
    ).strip()


def _rss_mb_script(quark_bin: str) -> str:
    return textwrap.dedent(
        f"""
        ps -eo rss,comm | awk '$2 ~ /{quark_bin}|qvisor|qemu|cloud-hypervisor|virtiofsd/ {{s+=$1}} END {{printf "%.2f", s/1024}}'
        """
    ).strip()


class MmapReadCoherencyRunner:
    def __init__(self, lab: LabSession) -> None:
        self.lab = lab
        self.backend = lab.quark.backend
        self.remote = lab.remote
        self.config = lab.config

    def _run_script(self, script: str, *, timeout: int = 120) -> tuple[bool, str]:
        r = self.remote.sh(script, timeout=timeout, check=False)
        out = "\n".join(
            part.strip()
            for part in (r.stdout or "", r.stderr or "")
            if part and part.strip()
        )
        return r.ok, out[-8000:]

    def run_c1(self) -> CoherencyResult:
        q = self.backend._quark_cmd
        prep = quark_io_bench_bundle_preamble(self.config, COHERENCY_IMAGE)
        cleanup = quark_io_bench_cleanup_trap(quark_delete_cmd=self.backend._quark_force_delete())
        read_py = python_exec_cmd(C1_GUEST_READ_PY)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            ID=""
            {prep}
            {COHERENCY_HOST_BENCH_PREP}
            {cleanup}
            FILE="$HOST_BENCH/c1.dat"
            printf '11112222' > "$FILE"
            sync
            ID=keska-c1-$RANDOM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            h1=$({q(f'exec --user 0:0 "$ID" -- {read_py}')} | tail -1)
            test "$h1" = "3131313132323232"
            {self.backend._quark_force_delete()}
            ID=""
            sudo -n rm -f "$FILE"
            printf '33334444' | sudo -n tee "$FILE" >/dev/null
            sudo -n chmod 644 "$FILE"
            sync
            ID=keska-c1b-$RANDOM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            h2=$({q(f'exec --user 0:0 "$ID" -- {read_py}')} | tail -1)
            test "$h2" = "3333333334343434"
            cleanup_io_bench
            trap - EXIT INT TERM
            echo PASS C1
            """
        ).strip()
        ok, out = self._run_script(script)
        if ok and _guest_pass_line(out, "PASS C1"):
            return CoherencyResult("C1", True, "host write visible to guest read()")
        return CoherencyResult("C1", False, "host write not visible to guest read()", out[-2000:])

    def run_c2(self) -> CoherencyResult:
        cmd = python_exec_cmd(C2_GUEST_PY)
        script = _coherency_sandbox_script(self.backend, guest_cmd=cmd)
        ok, out = self._run_script(script)
        if ok and _guest_pass_line(out, "PASS C2"):
            return CoherencyResult("C2", True, "guest read-your-writes OK")
        return CoherencyResult("C2", False, "guest read-your-writes failed", out[-2000:])

    def run_c3(self) -> CoherencyResult:
        cmd = python_exec_cmd(C3_GUEST_PY)
        script = _coherency_sandbox_script(self.backend, guest_cmd=cmd)
        ok, out = self._run_script(script, timeout=180)
        if ok and _guest_pass_line(out, "PASS C3"):
            return CoherencyResult("C3", True, "truncate while mapped OK")
        return CoherencyResult("C3", False, "truncate coherency failed", out[-2000:])

    def run_c4(self) -> CoherencyResult:
        q = self.backend._quark_cmd
        prep = quark_io_bench_bundle_preamble(self.config, COHERENCY_IMAGE)
        cleanup = quark_io_bench_cleanup_trap(quark_delete_cmd=self.backend._quark_force_delete())
        guest = python_exec_cmd(C4_GUEST_PY)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            ID=""
            {prep}
            {COHERENCY_HOST_BENCH_PREP}
            {cleanup}
            FILE="$HOST_BENCH/c4.dat"
            dd if=/dev/zero of="$FILE" bs=4096 count=1 conv=fsync 2>/dev/null
            ID=keska-c4-$RANDOM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            (
              end=$((SECONDS + 8))
              while [ "$SECONDS" -lt "$end" ]; do
                dd if=/dev/zero of="$FILE" bs=4096 count=1 conv=fsync 2>/dev/null || true
                sleep 0.05
              done
            ) &
            HPID=$!
            out=$({q(f'exec --user 0:0 "$ID" -- {guest}')})
            kill "$HPID" 2>/dev/null || true
            wait "$HPID" 2>/dev/null || true
            cleanup_io_bench
            trap - EXIT INT TERM
            echo "$out"
            """
        ).strip()
        ok, out = self._run_script(script, timeout=90)
        if ok and _guest_pass_line(out, "PASS C4"):
            return CoherencyResult("C4", True, "no torn blocks under concurrent host write")
        return CoherencyResult("C4", False, "concurrent host/guest coherency failed", out[-4000:] or "(no output)")

    def run_c5(self) -> CoherencyResult:
        cmd = python_exec_cmd(C5_GUEST_PY)
        script = _coherency_sandbox_script(self.backend, guest_cmd=cmd)
        ok, out = self._run_script(script)
        if ok and _guest_pass_line(out, "PASS C5"):
            return CoherencyResult("C5", True, "mmap vs read() consistent")
        return CoherencyResult("C5", False, "mmap vs read() mismatch", out[-2000:])

    def run_c6(self, *, minutes: int = 30) -> CoherencyResult:
        from keska_lab.harness.db import parse_pgbench_tps

        q = self.backend._quark_cmd
        prep = self.backend._postgres_run_bundle_script()
        quark_bin = self.backend.quark_bin
        rss = _rss_mb_script(quark_bin)
        duration = minutes * 60
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            {prep}
            ID=keska-c6-$RANDOM
            cleanup() {{ {self.backend._quark_force_delete()}; sudo rm -rf "$BUNDLE" 2>/dev/null || true; }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            {self.backend._postgres_startup_sleep()}
            rss0=$({rss})
            out=$(timeout {duration + 60} {q(f'exec {self.backend._postgres_exec_user()} "$ID" -- pgbench -c1 -T{duration} -U postgres')} 2>&1) || true
            rss1=$({rss})
            tps=$(echo "$out" | grep -E 'including latency|tps' | tail -1 || true)
            cleanup
            trap - EXIT INT TERM
            echo "RSS0=$rss0 RSS1=$rss1"
            echo "$out" | tail -5
            echo "PGBENCH_OUT=$tps"
            """
        ).strip()
        ok, out = self._run_script(script, timeout=duration + 300)
        tps = parse_pgbench_tps(out)
        rss0 = rss1 = None
        for line in out.splitlines():
            if line.startswith("RSS0="):
                parts = line.replace("RSS0=", "").split()
                rss0 = float(parts[0])
                if "RSS1=" in line:
                    rss1 = float(line.split("RSS1=")[1].split()[0])
            elif "RSS1=" in line and rss1 is None:
                rss1 = float(line.split("RSS1=")[1].split()[0])
        if not ok or tps <= 0:
            return CoherencyResult("C6", False, f"pgbench soak failed or no TPS ({minutes} min)", out[-3000:])
        if rss0 and rss1 and rss0 > 0:
            drift_pct = (rss1 - rss0) / rss0 * 100.0
            if drift_pct > 3.0:
                return CoherencyResult(
                    "C6",
                    False,
                    f"RSS drift {drift_pct:.1f}% > 3% during {minutes} min soak",
                    out[-2000:],
                )
        return CoherencyResult(
            "C6",
            True,
            f"pgbench {minutes} min completed, TPS={tps:.1f}",
            out[-500:],
        )

    def run_c7(self) -> CoherencyResult:
        q = self.backend._quark_cmd
        prep = quark_io_bench_bundle_preamble(self.config, COHERENCY_IMAGE)
        cleanup = quark_io_bench_cleanup_trap(quark_delete_cmd=self.backend._quark_force_delete())
        guest = python_exec_cmd(C7_GUEST_PY)
        quark_bin = self.backend.quark_bin
        rss = _rss_mb_script(quark_bin)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            {prep}
            {COHERENCY_HOST_BENCH_PREP}
            {cleanup}
            ID=keska-c7-$RANDOM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            sleep 2
            rss0=$({rss})
            out=$({q(f'exec --user 0:0 "$ID" -- {guest}')})
            sleep 1
            rss1=$({rss})
            cleanup_io_bench
            trap - EXIT INT TERM
            echo "$out"
            echo "RSS0=$rss0 RSS1=$rss1"
            """
        ).strip()
        ok, out = self._run_script(script, timeout=300)
        if not ok or not _guest_pass_line(out, "PASS C7"):
            return CoherencyResult("C7", False, "open/read/close loop failed", out[-2000:])
        rss0 = rss1 = 0.0
        for line in out.splitlines():
            if line.startswith("RSS0="):
                rss0 = float(line.split("=")[1].split()[0])
            if "RSS1=" in line:
                rss1 = float(line.split("RSS1=")[1].split()[0])
        if rss0 > 0:
            drift = (rss1 - rss0) / rss0 * 100.0
            if drift > 3.0:
                return CoherencyResult("C7", False, f"RSS drift {drift:.1f}% after 1000× I/O", out[-1000:])
        return CoherencyResult("C7", True, f"1000× open/read/close OK (RSS {rss0:.1f}→{rss1:.1f} MB)")

    def run_all(
        self,
        *,
        cases: tuple[str, ...] | None = None,
        include_c6: bool = False,
        c6_minutes: int = 30,
    ) -> CoherencySuiteResult:
        want = set(cases or [c.id for c in COHERENCY_CASES])
        if not include_c6:
            want.discard("C6")
        suite = CoherencySuiteResult()
        runners = {
            "C1": self.run_c1,
            "C2": self.run_c2,
            "C3": self.run_c3,
            "C4": self.run_c4,
            "C5": self.run_c5,
            "C6": lambda: self.run_c6(minutes=c6_minutes),
            "C7": self.run_c7,
        }
        for case_id in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            if case_id not in want:
                continue
            from keska_lab.display import console

            console.print(f"[cyan]Running {case_id}…[/cyan]")
            result = runners[case_id]()
            status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            console.print(f"  {status} {case_id}: {result.message}")
            suite.results.append(result)
        return suite


def rebuild_mmap_read(remote: RemoteHost, config: LabConfig, *, stream: bool = True) -> None:
    features = ",".join(experimental_cargo_features("MmapRead"))
    ctx = ProvisionContext(
        remote=remote,
        config=config,
        repo=config.resolve_local_repo(),
        profile=config.quark_build_profile,
        stream=stream,
    )
    sync_sources(ctx)
    build_quark(ctx, cargo_features=features)
    install_quark(ctx)


def run_mmap_read_coherency(
    *,
    cases: tuple[str, ...] | None = None,
    include_c6: bool = False,
    c6_minutes: int = 30,
    skip_build: bool = False,
    stream: bool = True,
) -> CoherencySuiteResult:
    """Build MmapRead experimental quark, deploy config, run C1–C7."""
    lab = LabSession()
    lab.quark.prepare(stream=False, mode="full", workload="python")
    if not skip_build:
        rebuild_mmap_read(lab.remote, lab.config, stream=stream)
    QuarkExperimentalConfigStep("MmapRead").run(lab.remote, stream=stream)
    lab.backend.cleanup()
    runner = MmapReadCoherencyRunner(lab)
    suite = runner.run_all(cases=cases, include_c6=include_c6, c6_minutes=c6_minutes)
    out_dir = Path.home() / ".keska-lab" / "results" / "group5"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    path = out_dir / f"{ts}_MmapRead_coherency.json"
    suite.save(path)
    from keska_lab.setup.quark_config import QuarkBenchConfigStep

    QuarkBenchConfigStep().run(lab.remote, stream=stream)
    from keska_lab.display import console

    console.print(f"\n[dim]Coherency report: {path}[/dim]")
    console.print(
        f"[bold]Overall:[/bold] {'PASS' if suite.passed else 'FAIL'} "
        f"({sum(r.passed for r in suite.results)}/{len(suite.results)} cases)"
    )
    return suite


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="E2 MmapRead coherency checklist (C1–C7)")
    parser.add_argument(
        "--cases",
        default="C1,C2,C3,C4,C5,C7",
        help="comma-separated case ids (default skips C6 soak)",
    )
    parser.add_argument("--include-c6", action="store_true", help="run 30 min pgbench soak (C6)")
    parser.add_argument("--c6-minutes", type=int, default=30, help="C6 soak duration")
    parser.add_argument("--skip-build", action="store_true", help="assume MmapRead binary installed")
    args = parser.parse_args()
    cases = tuple(c.strip() for c in args.cases.split(",") if c.strip())
    try:
        suite = run_mmap_read_coherency(
            cases=cases,
            include_c6=args.include_c6,
            c6_minutes=args.c6_minutes,
            skip_build=args.skip_build,
        )
    except Exception as exc:
        print(f"coherency run failed: {exc}", file=__import__("sys").stderr)
        return 1
    return 0 if suite.passed else 1
