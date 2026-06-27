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
