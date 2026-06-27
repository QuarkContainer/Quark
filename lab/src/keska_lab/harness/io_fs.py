"""Host-backed file I/O benchmark paths (real disk, not tmpfs)."""

from __future__ import annotations

import shlex
import textwrap

from keska_lab.config import LabConfig
from keska_lab.setup.oci_bundle import bundle_dir

# Guest mount point — bind-mounted from the lab host block filesystem.
IO_BENCH_MOUNT = "/bench"
IO_BENCH_FILE = f"{IO_BENCH_MOUNT}/keska-dd.img"
IO_BENCH_DD_COUNT = 64  # MiB
IO_CONCURRENT_READERS = 4


def io_bench_host_dir(config: LabConfig) -> str:
    """Directory on the lab host used for bind-mount I/O benchmarks."""
    return config.io_bench_dir.rstrip("/")


def dd_io_bench_sh(*, bench_file: str = IO_BENCH_FILE, count: int = IO_BENCH_DD_COUNT) -> str:
    """Shell snippet: dd write + read; prints WRITE/READ lines for parse_dd_mib_s."""
    bf = shlex.quote(bench_file)
    return (
        f"w=$(dd if=/dev/zero of={bf} bs=1M count={count} conv=fsync 2>&1 | tail -1); "
        f"r=$(dd if={bf} of=/dev/null bs=1M 2>&1 | tail -1); "
        f'echo WRITE \\"$w\\"; echo READ \\"$r\\"'
    )


def concurrent_read_bench_sh(
    *,
    bench_file: str = IO_BENCH_FILE,
    count: int = IO_BENCH_DD_COUNT,
    readers: int = IO_CONCURRENT_READERS,
) -> str:
    """Write file once, parallel dd reads (caller times the exec on the host)."""
    bf = shlex.quote(bench_file)
    return textwrap.dedent(
        f"""
        dd if=/dev/zero of={bf} bs=1M count={count} conv=fsync >/dev/null 2>&1
        for i in $(seq 1 {readers}); do
          dd if={bf} of=/dev/null bs=1M >/dev/null 2>&1 &
        done
        wait
        echo CONCURRENT_READ_OK
        """
    ).strip()


def concurrent_read_total_bytes(
    *,
    count: int = IO_BENCH_DD_COUNT,
    readers: int = IO_CONCURRENT_READERS,
) -> int:
    return count * readers * 1024 * 1024


def concurrent_read_mib_s(*, total_bytes: int, elapsed_ns: int) -> float:
    if elapsed_ns <= 0:
        return 0.0
    return total_bytes / (elapsed_ns / 1_000_000_000) / (1024 * 1024)


def parse_concurrent_read_mib_s(line: str) -> float:
    """Parse legacy CONCURRENT_READ bytes=N elapsed_ns=N lines."""
    parts = dict(
        item.split("=", 1)
        for item in line.removeprefix("CONCURRENT_READ ").split()
        if "=" in item
    )
    return concurrent_read_mib_s(
        total_bytes=int(parts["bytes"]),
        elapsed_ns=int(parts["elapsed_ns"]),
    )


def quark_io_bench_bundle_preamble(
    config: LabConfig,
    image: str,
    *,
    bundle_var: str = "BUNDLE",
) -> str:
    """
    Prepare a per-run OCI bundle with /bench bind-mounted from host disk.

    Sets BASE, HOST_BENCH, BUNDLE and patches config.json. Caller must trap cleanup.
    """
    base = bundle_dir(config, image)
    host_root = io_bench_host_dir(config)
    mount = IO_BENCH_MOUNT
    return textwrap.dedent(
        f"""
        BASE={shlex.quote(base)}
        HOST_BENCH={shlex.quote(host_root)}/run-$RANDOM
        {bundle_var}={shlex.quote(config.work_dir.rstrip("/"))}/io-run-$RANDOM
        sudo -n mkdir -p {shlex.quote(host_root)} "$HOST_BENCH"
        sudo -n chmod 755 "$HOST_BENCH"
        mkdir -p "${bundle_var}"
        ln -sfn "$BASE/rootfs" "${bundle_var}/rootfs"
        BASE="$BASE" BUNDLE="${bundle_var}" HOST_BENCH="$HOST_BENCH" MOUNT={shlex.quote(mount)} python3 <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ["BASE"])
bundle = pathlib.Path(os.environ["BUNDLE"])
host = os.environ["HOST_BENCH"]
mount = os.environ["MOUNT"]
cfg = json.loads((base / "config.json").read_text())
cfg["mounts"] = [m for m in cfg.get("mounts", []) if m.get("destination") != mount]
cfg["mounts"].append({{
    "destination": mount,
    "type": "bind",
    "source": host,
    "options": ["rbind", "rw"],
}})
(bundle / "config.json").write_text(json.dumps(cfg, indent=2) + "\\n")
PY
        """
    ).strip()


def quark_io_bench_cleanup_trap(*, quark_delete_cmd: str | None = None) -> str:
    delete = ""
    if quark_delete_cmd:
        delete = f"{quark_delete_cmd} >/dev/null 2>&1 || true\n  "
    return textwrap.dedent(
        f"""
        cleanup_io_bench() {{
          {delete}sudo -n rm -rf "$HOST_BENCH" 2>/dev/null || true
          rm -rf "$BUNDLE" 2>/dev/null || true
        }}
        trap cleanup_io_bench EXIT INT TERM
        """
    ).strip()


def kata_ctr_io_mount(host_bench: str) -> str:
    """containerd --mount flag for a host bind at /bench."""
    return f"type=bind,src={host_bench},dst={IO_BENCH_MOUNT},options=rbind:rw"
