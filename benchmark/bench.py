#!/usr/bin/env python3
"""
Quark benchmark orchestrator — runs inside Lima VM via vm-exec.sh from the macOS host.

Usage:
  python3 benchmark/bench.py run [--profile dev|stress] [--suite all|cold_start|...]
  python3 benchmark/bench.py setup
  python3 benchmark/bench.py results
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
VM_EXEC = REPO_ROOT / "scripts" / "dev" / "bin" / "vm-exec.sh"
RESULTS_DIR = BENCH_DIR / "results"
BENCH_IPERF_IMAGE = "quark-bench-iperf"
NETWORK_NAME = "quark-bench-net"
MEM_MIN_KB = 512_000  # abort if MemAvailable drops below ~500 MiB

PROFILES = {
    "dev": {"wave_size": 20, "waves": 5, "hibernate_n": 5, "memory_n": 5},
    "stress": {"wave_size": 50, "waves": 20, "hibernate_n": 20, "memory_n": 20},
}

ALL_SUITES = (
    "cold_start",
    "hibernate",
    "memory",
    "network_cluster",
    "network_inet",
    "io_fs",
)


@dataclass
class Config:
    runtime: str = "quark_d"
    profile: str = "dev"
    suite: str = "all"
    image: str = "ubuntu:24.04"
    output: Path | None = None
    wave_size: int | None = None
    waves: int | None = None
    stagger_ms: int = 50
    verbose: bool = False


@dataclass
class VmExec:
    env: dict[str, str] = field(default_factory=dict)

    def run(
        self,
        *args: str,
        check: bool = True,
        capture: bool = True,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if not VM_EXEC.is_file():
            raise SystemExit(f"vm-exec not found: {VM_EXEC}")
        cmd = [str(VM_EXEC), *args]
        env = {**os.environ, **self.env}
        return subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
            env=env,
            timeout=timeout,
        )

    def bash(self, script: str, check: bool = True, timeout: int | None = None) -> str:
        proc = self.run("bash", "-c", script, check=check, timeout=timeout)
        return proc.stdout if proc.stdout else ""

    def docker(self, *args: str, **kwargs: Any) -> str:
        return self.run("docker", *args, **kwargs).stdout or ""


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"samples": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(math.ceil(p / 100.0 * n) - 1)))
        return s[idx]

    return {
        "samples": n,
        "mean": round(statistics.mean(s), 2),
        "p50": round(pct(50), 2),
        "p95": round(pct(95), 2),
        "p99": round(pct(99), 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "stdev": round(statistics.stdev(s), 2) if n > 1 else 0.0,
    }


def print_stats_table(title: str, unit: str, st: dict[str, Any]) -> None:
    print(f"\n  {title} ({unit})")
    print(f"    samples={st['samples']}  mean={st['mean']}  p50={st['p50']}  "
          f"p95={st['p95']}  p99={st['p99']}  min={st['min']}  max={st['max']}")


def host_snapshot(vm: VmExec, phase: str) -> dict[str, Any]:
    out = vm.bash(
        "cat /proc/loadavg; awk '/MemAvailable/ {print $2}' /proc/meminfo; "
        "awk '/MemTotal/ {print $2}' /proc/meminfo"
    )
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    load = [float(x) for x in lines[0].split()[:3]] if lines else [0, 0, 0]
    mem_avail = int(lines[1]) if len(lines) > 1 else 0
    mem_total = int(lines[2]) if len(lines) > 2 else 0
    return {
        "phase": phase,
        "load_avg": load,
        "mem_available_kb": mem_avail,
        "mem_total_kb": mem_total,
    }


def quark_rss_mb(vm: VmExec) -> float:
    script = r"""
total=0
for pid in $(pgrep -x quark_d 2>/dev/null; pgrep -f '/usr/local/bin/quark' 2>/dev/null | sort -u); do
  rss=$(awk '/VmRSS/ {print $2}' /proc/$pid/status 2>/dev/null || echo 0)
  total=$((total + rss))
done
echo $total
"""
    raw = vm.bash(script).strip()
    line = raw.splitlines()[-1] if raw else "0"
    try:
        kb = int(re.sub(r"\D", "", line) or "0")
    except ValueError:
        kb = 0
    return round(kb / 1024.0, 2)


def mem_available_kb(vm: VmExec) -> int:
    out = vm.bash("awk '/MemAvailable/ {print $2}' /proc/meminfo").strip()
    try:
        return int(out.splitlines()[-1])
    except (ValueError, IndexError):
        return 0


def check_mem(vm: VmExec) -> None:
    avail = mem_available_kb(vm)
    if avail and avail < MEM_MIN_KB:
        raise SystemExit(
            f"MemAvailable {avail} KiB below threshold {MEM_MIN_KB}; aborting benchmark."
        )


def collect_metadata(vm: VmExec, cfg: Config) -> dict[str, Any]:
    prof = PROFILES[cfg.profile]
    wave_size = cfg.wave_size or prof["wave_size"]
    waves = cfg.waves or prof["waves"]
    arch = vm.bash("uname -m").strip()
    cpus = vm.bash("nproc").strip()
    quark_info = vm.bash(
        "ls -l /usr/local/bin/quark_d /usr/local/bin/qkernel_d.bin 2>/dev/null || true"
    ).strip()
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "runtime": cfg.runtime,
        "profile": cfg.profile,
        "image": cfg.image,
        "lima": {
            "instance": os.environ.get("LIMA_INSTANCE", "quark"),
            "cpus": int(cpus) if cpus.isdigit() else cpus,
            "arch": arch,
        },
        "quark": {"binaries": quark_info},
        "params": {
            "wave_size": wave_size,
            "waves": waves,
            "hibernate_n": prof["hibernate_n"],
            "memory_n": prof["memory_n"],
        },
    }


def preflight(vm: VmExec, cfg: Config) -> None:
    print("→ Preflight…", flush=True)
    vm.run("docker", "info", check=True)
    out = vm.docker("info")
    if cfg.runtime not in out and "quark" not in out:
        print(f"  warning: runtime {cfg.runtime} may not be registered", file=sys.stderr)
    check_mem(vm)
    orphans = quark_orphan_count(vm)
    load = vm_load_avg(vm)
    if orphans > 0:
        print(f"  warning: {orphans} orphaned quark_d process(es) — run: python3 benchmark/bench.py cleanup", flush=True)
    if load[0] > vm_cpu_count(vm) * 2:
        print(f"  warning: high load avg {load[0]:.1f} (VM may be slow until cleaned up)", flush=True)
    print("  ✓ VM docker OK", flush=True)


def vm_cpu_count(vm: VmExec) -> int:
    out = vm.bash("nproc").strip()
    return int(out) if out.isdigit() else 8


def vm_load_avg(vm: VmExec) -> tuple[float, float, float]:
    out = vm.bash("awk '{print $1,$2,$3}' /proc/loadavg").strip()
    parts = out.split()
    if len(parts) >= 3:
        return float(parts[0]), float(parts[1]), float(parts[2])
    return 0.0, 0.0, 0.0


def quark_orphan_count(vm: VmExec) -> int:
    """Count quark_d processes when no docker containers are running."""
    out = vm.bash(
        "running=$(docker ps -q | wc -l); "
        "orphans=$(pgrep -c quark_d 2>/dev/null || echo 0); "
        "if [ \"$running\" -eq 0 ]; then echo \"$orphans\"; else echo 0; fi",
        check=False,
    ).strip()
    try:
        return int(out.splitlines()[-1])
    except ValueError:
        return 0


def cleanup_quark_orphans(vm: VmExec, *, restart_docker: bool = False) -> None:
    """Kill quark_d processes left behind when no containers are running."""
    before = quark_orphan_count(vm)
    if before == 0 and not restart_docker:
        return
    if before > 0:
        print(f"  → cleaning {before} orphaned quark_d process(es)…", flush=True)
        vm.bash("sudo killall -9 quark_d 2>/dev/null || true", check=False)
        time.sleep(1)
    if restart_docker or before > 0:
        vm.bash("sudo systemctl restart docker", check=False)
        time.sleep(2)
    after = quark_orphan_count(vm)
    if before > 0:
        print(f"  ✓ quark_d orphans: {before} → {after}", flush=True)


def cleanup_network(vm: VmExec) -> None:
    vm.bash(
        "docker rm -f iperf-s iperf-c 2>/dev/null; "
        f"docker network rm {NETWORK_NAME} 2>/dev/null; "
        "docker rm -f $(docker ps -aq --filter name=bench-) 2>/dev/null || true",
        check=False,
    )


def ensure_network(vm: VmExec) -> None:
    vm.bash(
        f"docker network inspect {NETWORK_NAME} >/dev/null 2>&1 || "
        f"docker network create {NETWORK_NAME}",
        check=True,
    )


# --- Suites ---


def suite_cold_start(vm: VmExec, cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    wave_size = meta["params"]["wave_size"]
    waves = meta["params"]["waves"]
    runtime = cfg.runtime
    image = cfg.image
    stagger_ms = cfg.stagger_ms
    print(f"\n== cold_start (waves={waves} × {wave_size}, runtime={runtime}, image={image}) ==")

    cleanup_quark_orphans(vm)

    # Warmup — discard first-run image/layer cost from stats
    print("  warmup…")
    warmup_err = vm.bash(
        f"docker run --runtime={runtime} --rm {image} /bin/true",
        check=False,
        timeout=120,
    )
    if warmup_err.strip():
        print(f"  warning: warmup failed: {warmup_err.strip()[:200]}", flush=True)

    all_ms: list[float] = []
    wave_results: list[dict[str, Any]] = []
    all_errors: list[str] = []

    for wave in range(1, waves + 1):
        check_mem(vm)
        cleanup_quark_orphans(vm)
        script = f"""
RUNTIME={runtime!r}
IMAGE={image!r}
WAVE={wave_size}
STAGGER_MS={stagger_ms}
DIR=$(mktemp -d)
for i in $(seq 1 $WAVE); do
  (
    msfile="$DIR/$i.ms"
    errfile="$DIR/$i.err"
    t0=$(date +%s%N)
    if docker run --runtime="$RUNTIME" --rm "$IMAGE" /bin/true >/dev/null 2>"$errfile"; then
      t1=$(date +%s%N)
      echo $(( (t1 - t0) / 1000000 )) > "$msfile"
    else
      echo "exit:$?" >> "$errfile"
    fi
  ) &
  if [ "$STAGGER_MS" -gt 0 ]; then
    sleep "$(awk "BEGIN {{printf \\"%.3f\\", $STAGGER_MS/1000}}")"
  fi
done
wait
echo "---MS---"
cat "$DIR"/*.ms 2>/dev/null || true
echo "---ERR---"
for f in "$DIR"/*.err; do
  [ -s "$f" ] || continue
  echo "job$(basename "$f" .err):$(tr '\\n' ' ' < "$f" | head -c 200)"
done
rm -rf "$DIR"
"""
        out = vm.bash(script, timeout=600)
        ms_section = out.split("---MS---")[-1].split("---ERR---")[0] if "---MS---" in out else out
        err_section = out.split("---ERR---")[-1] if "---ERR---" in out else ""
        wave_ms = [float(x) for x in ms_section.split() if x.strip().isdigit()]
        wave_errors = [ln.strip() for ln in err_section.splitlines() if ln.strip()]
        failed = wave_size - len(wave_ms)
        if wave_errors:
            all_errors.extend(wave_errors)
            if cfg.verbose or failed:
                for err in wave_errors[:3]:
                    print(f"  error: {err}", flush=True)
        if failed:
            print(f"  warning: {failed} containers failed in wave {wave}", flush=True)
            vm.bash("sleep 2", check=False)
        all_ms.extend(wave_ms)
        st = stats(wave_ms)
        wave_results.append({
            "wave": wave,
            "started": wave_size,
            "failed": failed,
            "errors": wave_errors[:5],
            "stats": st,
        })
        print(f"  wave {wave}/{waves}: {len(wave_ms)} ok, mean={st['mean']} ms")
        time.sleep(1)

    cleanup_quark_orphans(vm)
    st_all = stats(all_ms)
    print_stats_table("cold_start total", "ms", st_all)
    result: dict[str, Any] = {
        "unit": "ms",
        "stats": st_all,
        "waves": wave_results,
        "samples": len(all_ms),
    }
    if all_errors:
        result["errors_sample"] = all_errors[:10]
    orphans = quark_orphan_count(vm)
    if orphans:
        result["orphan_quark_d"] = orphans
    return result


def suite_hibernate(vm: VmExec, cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    n = meta["params"]["hibernate_n"]
    runtime = cfg.runtime
    print(f"\n== hibernate (n={n}, runtime={runtime}) ==")

    names = " ".join(f"bench-hiber-{i}" for i in range(n))
    script = f"""
set -e
RUNTIME={runtime!r}
N={n}
for i in $(seq 0 $((N-1))); do
  name=bench-hiber-$i
  docker rm -f "$name" 2>/dev/null || true
  docker run -d --name "$name" --runtime="$RUNTIME" busybox sleep 3600
done
sleep 2
rss_before=$( (for pid in $(pgrep -x quark_d 2>/dev/null; pgrep -f /usr/local/bin/quark 2>/dev/null | sort -u); do
  awk '/VmRSS/ {{print $2}}' /proc/$pid/status 2>/dev/null; done) | awk '{{s+=$1}} END {{print s+0}}')
for i in $(seq 0 $((N-1))); do
  name=bench-hiber-$i
  t0=$(date +%s%N)
  docker pause "$name"
  t1=$(date +%s%N)
  echo PAUSE $(( (t1 - t0) / 1000000 ))
done
sleep 5
rss_after=$( (for pid in $(pgrep -x quark_d 2>/dev/null; pgrep -f /usr/local/bin/quark 2>/dev/null | sort -u); do
  awk '/VmRSS/ {{print $2}}' /proc/$pid/status 2>/dev/null; done) | awk '{{s+=$1}} END {{print s+0}}')
for i in $(seq 0 $((N-1))); do
  name=bench-hiber-$i
  t0=$(date +%s%N)
  docker unpause "$name"
  t1=$(date +%s%N)
  echo RESUME $(( (t1 - t0) / 1000000 ))
done
echo RSS_BEFORE $rss_before
echo RSS_AFTER $rss_after
for i in $(seq 0 $((N-1))); do docker rm -f bench-hiber-$i 2>/dev/null || true; done
"""
    out = vm.bash(script, timeout=300)
    pause_latencies: list[float] = []
    resume_latencies: list[float] = []
    rss_before_kb = rss_after_kb = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "PAUSE":
            pause_latencies.append(float(parts[1]))
        elif len(parts) == 2 and parts[0] == "RESUME":
            resume_latencies.append(float(parts[1]))
        elif len(parts) == 2 and parts[0] == "RSS_BEFORE":
            rss_before_kb = int(parts[1])
        elif len(parts) == 2 and parts[0] == "RSS_AFTER":
            rss_after_kb = int(parts[1])

    rss_before = round(rss_before_kb / 1024.0, 2)
    rss_after = round(rss_after_kb / 1024.0, 2)
    result = {
        "pause_latency_ms": {"stats": stats(pause_latencies)},
        "resume_latency_ms": {"stats": stats(resume_latencies)},
        "rss_mb": {
            "before": rss_before,
            "after_pause": rss_after,
            "delta_pause": round(rss_after - rss_before, 2),
        },
        "containers": n,
    }
    print(f"  RSS: {rss_before} → {rss_after} MB (Δ {result['rss_mb']['delta_pause']} MB)")
    print_stats_table("pause", "ms", result["pause_latency_ms"]["stats"])
    print_stats_table("resume", "ms", result["resume_latency_ms"]["stats"])
    return result


def suite_memory(vm: VmExec, cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    n = meta["params"]["memory_n"]
    runtime = cfg.runtime
    print(f"\n== memory (n={n}, runtime={runtime}) ==", flush=True)

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            return _suite_memory_once(vm, cfg, n, runtime)
        except subprocess.CalledProcessError as e:
            last_err = e
            vm.bash("sleep 3", check=False)
            vm.bash("docker rm -f $(docker ps -aq --filter name=bench-mem-) 2>/dev/null || true", check=False)
    raise last_err  # type: ignore[misc]


def _suite_memory_once(vm: VmExec, cfg: Config, n: int, runtime: str) -> dict[str, Any]:
    vm.bash("docker rm -f $(docker ps -aq --filter name=bench-mem-) 2>/dev/null || true", check=False)
    rss0 = quark_rss_mb(vm)
    snap0 = host_snapshot(vm, "baseline")

    names = []
    for i in range(n):
        name = f"bench-mem-{i}"
        names.append(name)
        vm.bash(f"docker run -d --name {name} --runtime={runtime} busybox sleep 3600", timeout=60)

    time.sleep(2)
    rss_idle = quark_rss_mb(vm)
    snap_idle = host_snapshot(vm, "idle")

    per_container = round((rss_idle - rss0) / max(n, 1), 2)

    loaded_name = "bench-mem-load"
    vm.bash(f"docker rm -f {loaded_name} 2>/dev/null || true", check=False)
    vm.bash(
        f"docker run -d --name {loaded_name} --runtime={runtime} busybox sh -c "
        f"'while true; do :; done'",
        timeout=60,
    )
    time.sleep(2)
    rss_loaded = quark_rss_mb(vm)
    snap_loaded = host_snapshot(vm, "loaded")

    for name in names + [loaded_name]:
        vm.bash(f"docker rm -f {name} 2>/dev/null || true", check=False)

    result = {
        "idle_per_container_mb": {"stats": {"samples": n, "mean": per_container}},
        "rss_mb": {
            "baseline": rss0,
            "idle_total": rss_idle,
            "loaded_total": rss_loaded,
            "delta_idle": round(rss_idle - rss0, 2),
            "delta_loaded": round(rss_loaded - rss0, 2),
        },
        "containers_idle": n,
    }
    print(f"  idle overhead ~{per_container} MB/container (total Δ {result['rss_mb']['delta_idle']} MB)")
    print(f"  loaded total RSS ~{rss_loaded} MB")
    return result


def suite_network_cluster(vm: VmExec, cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    runtime = cfg.runtime
    print(f"\n== network_cluster (runtime={runtime}, {BENCH_IPERF_IMAGE}) ==")

    vm.bash(f"docker network rm {NETWORK_NAME} 2>/dev/null || true", check=False)
    ensure_network(vm)
    vm.bash("docker rm -f iperf-s 2>/dev/null || true", check=False)

    # Build image on first use if missing (also done in bench setup)
    vm.bash(
        f"docker image inspect {BENCH_IPERF_IMAGE} >/dev/null 2>&1 || "
        f"(printf 'FROM alpine\\nRUN apk add --no-cache iperf3\\n' | "
        f"docker build -t {BENCH_IPERF_IMAGE} -)",
        timeout=600,
        check=False,
    )

    vm.bash(
        f"docker run -d --name iperf-s --network {NETWORK_NAME} --runtime={runtime} "
        f"{BENCH_IPERF_IMAGE} iperf3 -s",
        timeout=120,
    )
    time.sleep(2)

    throughputs: list[float] = []
    for run in range(3):
        out = vm.bash(
            f"docker run --rm --network {NETWORK_NAME} --runtime={runtime} "
            f"{BENCH_IPERF_IMAGE} iperf3 -c iperf-s -t 5 -f m 2>&1",
            timeout=120,
            check=False,
        )
        found = False
        for line in out.splitlines():
            m = re.search(r"([\d.]+)\s+Mbits/sec", line)
            if not m:
                continue
            if "receiver" in line.lower() or not found:
                throughputs.append(float(m.group(1)))
                found = True
                break
        print(f"  run {run + 1}: {throughputs[-1] if throughputs else 'n/a'} Mbits/sec")

    vm.bash("docker rm -f iperf-s 2>/dev/null || true", check=False)
    vm.bash(f"docker network rm {NETWORK_NAME} 2>/dev/null || true", check=False)

    if not throughputs:
        return {"throughput_mbps": {"stats": stats([])}, "runs": 0, "skipped": True, "error": "no iperf results"}

    st = stats(throughputs)
    print_stats_table("throughput", "Mbits/sec", st)
    return {
        "throughput_mbps": {"stats": st, "unit": "Mbits/sec"},
        "runs": len(throughputs),
        "skipped": False,
    }


def docker_python(vm: VmExec, runtime: str, code: str, timeout: int = 90) -> str:
    """Run Python code inside a container (avoids shell quoting issues)."""
    encoded = base64.b64encode(code.encode()).decode()
    return vm.bash(
        f"docker run --rm --runtime={runtime} python:3.12-slim python3 -c "
        f"\"import base64; exec(base64.b64decode('{encoded}').decode())\"",
        timeout=timeout,
        check=False,
    )


def suite_network_inet(vm: VmExec, cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    runtime = cfg.runtime
    print(f"\n== network_inet (runtime={runtime}) ==")

    connect_ms: list[float] = []
    targets = [("1.1.1.1", 443), ("cloudflare.com", 443)]

    for host, port in targets:
        code = f"""
import socket, time
host, port = {host!r}, {port}
t0 = time.perf_counter()
s = socket.create_connection((host, port), timeout=15)
s.close()
print(int((time.perf_counter() - t0) * 1000))
"""
        out = docker_python(vm, runtime, code, timeout=60).strip()
        line = out.splitlines()[-1] if out else ""
        if line.isdigit():
            ms = float(line)
            connect_ms.append(ms)
            print(f"  {host}:{port} connect={ms:.0f} ms")
        else:
            print(f"  {host}:{port} skipped")

    download_mbps: list[float] = []
    dl_code = """
import time, urllib.request
url = 'https://speed.cloudflare.com/__down?bytes=5000000'
t0 = time.perf_counter()
data = urllib.request.urlopen(url, timeout=30).read()
elapsed = time.perf_counter() - t0
print(len(data) * 8 / elapsed / 1e6)
"""
    out = docker_python(vm, runtime, dl_code, timeout=90).strip()
    line = out.splitlines()[-1] if out else ""
    try:
        mbps = float(line)
        if mbps > 0:
            download_mbps.append(mbps)
            print(f"  download ~{mbps:.1f} Mbits/sec (cloudflare 5MB)")
    except ValueError:
        print("  download skipped")

    return {
        "tcp_connect_ms": {"stats": stats(connect_ms), "unit": "ms"},
        "download_mbps": {"stats": stats(download_mbps), "unit": "Mbits/sec"},
        "skipped": len(connect_ms) == 0 and len(download_mbps) == 0,
    }


def suite_io_fs(vm: VmExec, cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    runtime = cfg.runtime
    print(f"\n== io_fs (runtime={runtime}, busybox dd) ==")

    script = f"""
docker run --rm --runtime={runtime} busybox sh -c '
  wline=$(dd if=/dev/zero of=/tmp/bench bs=1M count=64 conv=fsync 2>&1 | tail -1)
  rline=$(dd if=/tmp/bench of=/dev/null bs=1M 2>&1 | tail -1)
  echo WRITE "$wline"
  echo READ "$rline"
'
"""
    out = vm.bash(script, timeout=120)

    def parse_mibs(line: str) -> float:
        m = re.search(r"([\d.]+)\s*([MG])B/s", line, re.I)
        if not m:
            return 0.0
        val = float(m.group(1))
        if m.group(2).upper() == "G":
            val *= 1024
        return round(val, 2)

    write_mib_s = read_mib_s = 0.0
    for line in out.splitlines():
        if line.startswith("WRITE "):
            write_mib_s = parse_mibs(line)
        elif line.startswith("READ "):
            read_mib_s = parse_mibs(line)

    print(f"  write {write_mib_s} MiB/s")
    print(f"  read  {read_mib_s} MiB/s")

    return {
        "method": "dd",
        "write_mib_s": write_mib_s,
        "read_mib_s": read_mib_s,
        "size_mib": 64,
    }


SUITE_FUNCS = {
    "cold_start": suite_cold_start,
    "hibernate": suite_hibernate,
    "memory": suite_memory,
    "network_cluster": suite_network_cluster,
    "network_inet": suite_network_inet,
    "io_fs": suite_io_fs,
}


def run_benchmark(cfg: Config) -> Path:
    vm = VmExec()
    preflight(vm, cfg)
    cleanup_network(vm)

    meta = collect_metadata(vm, cfg)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "metadata": meta,
        "suites": {},
        "host_snapshots": [host_snapshot(vm, "start")],
    }

    suites = ALL_SUITES if cfg.suite == "all" else [cfg.suite]
    for name in suites:
        if name not in SUITE_FUNCS:
            raise SystemExit(f"unknown suite: {name}")
        cleanup_network(vm)
        time.sleep(2)
        try:
            report["suites"][name] = SUITE_FUNCS[name](vm, cfg, meta)
        except subprocess.TimeoutExpired:
            report["suites"][name] = {"error": "timeout", "skipped": True}
            print(f"  ✗ {name} timed out", file=sys.stderr, flush=True)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or str(e)).strip()
            report["suites"][name] = {"error": err, "skipped": True}
            print(f"  ✗ {name} failed", file=sys.stderr, flush=True)
            if cfg.verbose:
                print(err, file=sys.stderr)

    cleanup_network(vm)
    report["host_snapshots"].append(host_snapshot(vm, "end"))

    ts = meta["timestamp"].replace(":", "-")
    sha = meta["git_commit"]
    out_path = cfg.output or (RESULTS_DIR / f"{ts}_{sha}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\n" + "=" * 60)
    print(f"✓ Report written: {out_path}")
    print("=" * 60)
    return out_path


def cmd_cleanup(_: argparse.Namespace) -> None:
    vm = VmExec()
    cleanup_network(vm)
    cleanup_quark_orphans(vm, restart_docker=True)
    load = vm_load_avg(vm)
    print(f"  load avg: {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}", flush=True)


def cmd_setup(_: argparse.Namespace) -> None:
    vm = VmExec()
    print("→ Installing benchmark tools in VM…", flush=True)
    vm.bash(
        "sudo apt-get update -qq && "
        "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fio iperf3 curl 2>/dev/null || "
        "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y fio iperf3 curl",
        timeout=600,
    )
    for img in ("alpine", "busybox", "python:3.12-slim", "ubuntu:24.04"):
        print(f"  pulling {img}…", flush=True)
        vm.bash(f"docker pull {img} 2>/dev/null | tail -1", check=False, timeout=300)
    print(f"  building {BENCH_IPERF_IMAGE} image…", flush=True)
    vm.bash(
        f"printf 'FROM alpine\\nRUN apk add --no-cache iperf3\\n' | "
        f"docker build -t {BENCH_IPERF_IMAGE} -",
        timeout=600,
        check=False,
    )
    print("  ✓ fio, iperf3, curl + docker images")


def cmd_results(_: argparse.Namespace) -> None:
    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("No results in benchmark/results/")
        return
    for p in files[:20]:
        print(f"  {p.name}  ({p.stat().st_size // 1024} KiB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quark benchmark suite")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run benchmark suite")
    run_p.add_argument("--profile", choices=list(PROFILES), default="dev")
    run_p.add_argument("--suite", default="all", choices=["all", *ALL_SUITES])
    run_p.add_argument("--runtime", default=os.environ.get("RUNTIME", "quark_d"))
    run_p.add_argument("--image", default="ubuntu:24.04")
    run_p.add_argument("--wave-size", type=int, default=None)
    run_p.add_argument("--waves", type=int, default=None)
    run_p.add_argument("--stagger-ms", type=int, default=50, help="Delay between parallel starts (0=none)")
    run_p.add_argument("--output", type=Path, default=None)
    run_p.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("setup", help="Install fio/iperf3 in VM")
    sub.add_parser("cleanup", help="Kill orphaned quark_d and restart docker")
    sub.add_parser("results", help="List recent result files")

    args = parser.parse_args(argv)

    if args.command == "setup":
        cmd_setup(args)
        return 0
    if args.command == "cleanup":
        cmd_cleanup(args)
        return 0
    if args.command == "results":
        cmd_results(args)
        return 0
    if args.command == "run":
        cfg = Config(
            runtime=args.runtime,
            profile=args.profile,
            suite=args.suite,
            image=args.image,
            output=args.output,
            wave_size=args.wave_size,
            waves=args.waves,
            stagger_ms=args.stagger_ms,
            verbose=args.verbose,
        )
        run_benchmark(cfg)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
