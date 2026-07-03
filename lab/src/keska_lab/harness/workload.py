"""Benchmark workload definitions (images + exec probes)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    image: str
    tti_exec: str = "/bin/echo ok"
    idle_cmd: str = "/bin/sleep 600"
    oci_args: tuple[str, ...] = ("/bin/sleep", "3600")
    oci_env: tuple[str, ...] = ("PATH=/usr/sbin:/usr/bin:/sbin:/bin",)
    oci_mounts: tuple[dict, ...] = ()
    oci_user: tuple[int, int] | None = None
    tti_timeout: int = 300
    cpu_loop_iters: int = 2_000_000
    # Inner body for `/bin/sh -c` (not a full shell command — avoids $ expansion in harness scripts).
    cpu_loop_cmd: str = "i=0; while [ $i -lt 2000000 ]; do i=$((i+1)); done; echo OK"


def cpu_loop_inner(spec: WorkloadSpec) -> str:
    """Return the sh -c script body for the CPU loop benchmark."""
    return normalize_cpu_loop_inner(spec.cpu_loop_cmd, iters=spec.cpu_loop_iters)


def normalize_cpu_loop_inner(exec_cmd: str | None = None, *, iters: int = 2_000_000) -> str:
    """Normalize legacy full commands to an inner sh body safe for harness embedding."""
    if not exec_cmd:
        return f"i=0; while [ $i -lt {iters} ]; do i=$((i+1)); done; echo OK"
    cmd = exec_cmd.strip()
    if cmd.startswith("/bin/sh -c "):
        rest = cmd[len("/bin/sh -c ") :].strip()
        if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in "\"'":
            cmd = rest[1:-1]
    if "2000000" in cmd and iters != 2_000_000:
        cmd = cmd.replace("2000000", str(iters), 1)
    return cmd


WORKLOADS: dict[str, WorkloadSpec] = {
    "busybox": WorkloadSpec(
        name="busybox",
        image="busybox",
        tti_exec="/bin/echo ok",
    ),
    "python": WorkloadSpec(
        name="python",
        image="python:3.12-slim",
        tti_exec="python3 -c \"print('ok')\"",
        idle_cmd="python3 -c \"import time; time.sleep(600)\"",
        oci_args=("/usr/local/bin/python3", "-c", "import time; time.sleep(3600)"),
        oci_env=(
            "PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONUNBUFFERED=1",
        ),
    ),
    "postgres": WorkloadSpec(
        name="postgres",
        image="postgres:16-alpine",
        tti_exec="pg_isready -U postgres",
        idle_cmd="docker-entrypoint.sh postgres",
        oci_args=("postgres", "-D", "/var/lib/postgresql/data"),
        oci_env=(
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PGDATA=/var/lib/postgresql/data",
            "POSTGRES_PASSWORD=bench",
            "POSTGRES_HOST_AUTH_METHOD=trust",
            "LANG=en_US.utf8",
        ),
        oci_mounts=(
            {
                "destination": "/dev/shm",
                "type": "tmpfs",
                "source": "shm",
                "options": ["nosuid", "nodev", "noexec", "mode=1777", "size=64m"],
            },
            {
                "destination": "/var/run/postgresql",
                "type": "tmpfs",
                "source": "pg_run",
                "options": ["nosuid", "nodev", "mode=1777", "size=8m"],
            },
        ),
        oci_user=(70, 70),
        tti_timeout=180,
    ),
    "iperf": WorkloadSpec(
        name="iperf",
        image="networkstatic/iperf3",
        tti_exec="iperf3 --version",
        idle_cmd="iperf3 -s",
        oci_args=("/bin/sh", "-c", "iperf3 -s"),
    ),
}


def get_workload(name: str) -> WorkloadSpec:
    key = name.lower().strip()
    if key not in WORKLOADS:
        known = ", ".join(sorted(WORKLOADS))
        raise ValueError(f"unknown workload {name!r}; choose from: {known}")
    return WORKLOADS[key]


def workload_images(names: list[str] | None = None) -> list[str]:
    if not names:
        return [w.image for w in WORKLOADS.values()]
    return [get_workload(n).image for n in names]


def is_postgres_workload(image: str, exec_cmd: str = "") -> bool:
    return "postgres" in image.lower() or "pg_isready" in exec_cmd


# Self-timed Python micro-benchmarks (stdout: one numeric line).
MICRO_GETPID_PY = """
import os, time
t = time.monotonic()
for _ in range(100_000):
    os.getpid()
print(int((time.monotonic() - t) * 1e9 / 100_000))
""".strip()

MICRO_MMAP_FAULT_PY = """
import mmap, time
SIZE = 64 * 1024 * 1024
t = time.monotonic()
m = mmap.mmap(-1, SIZE)
m.write(b"x" * SIZE)
m.close()
print(f"{(time.monotonic() - t) * 1000:.1f}")
""".strip()

MICRO_PIPE_IPC_PY = """
import os, time, threading
r, w = os.pipe()
CHUNK = 64 * 1024
N = 512
def writer():
    buf = b"x" * CHUNK
    for _ in range(N):
        os.write(w, buf)
    os.close(w)
t0 = time.monotonic()
t = threading.Thread(target=writer)
t.start()
received = 0
while True:
    chunk = os.read(r, CHUNK)
    if not chunk:
        break
    received += len(chunk)
t.join()
os.close(r)
elapsed = time.monotonic() - t0
print(f"{received / elapsed / (1024 * 1024):.2f}")
""".strip()

MICRO_BENCH_SCRIPTS: dict[str, str] = {
    "getpid_ns": MICRO_GETPID_PY,
    "mmap_anon_fault_ms": MICRO_MMAP_FAULT_PY,
    "pipe_throughput_mib_s": MICRO_PIPE_IPC_PY,
}


def micro_bench_script(metric: str) -> str:
    key = metric.strip()
    if key not in MICRO_BENCH_SCRIPTS:
        known = ", ".join(sorted(MICRO_BENCH_SCRIPTS))
        raise ValueError(f"unknown micro metric {metric!r}; choose from: {known}")
    return MICRO_BENCH_SCRIPTS[key]
