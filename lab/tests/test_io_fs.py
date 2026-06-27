"""Tests for host-backed I/O benchmark helpers."""

from __future__ import annotations

from keska_lab.config import LabConfig
from keska_lab.harness.io_fs import (
    IO_BENCH_FILE,
    IO_BENCH_MOUNT,
    concurrent_read_bench_sh,
    dd_io_bench_sh,
    io_bench_host_dir,
    quark_io_bench_bundle_preamble,
)


def test_io_bench_defaults_to_var_lib() -> None:
    cfg = LabConfig()
    assert io_bench_host_dir(cfg) == "/var/lib/keska-lab/io-bench"


def test_io_bench_mount_and_file() -> None:
    assert IO_BENCH_MOUNT == "/bench"
    assert IO_BENCH_FILE == "/bench/keska-dd.img"


def test_dd_uses_bench_mount_not_tmp() -> None:
    sh = dd_io_bench_sh()
    assert "/bench/" in sh
    assert "/tmp/" not in sh


def test_quark_preamble_mentions_bind_mount() -> None:
    cfg = LabConfig()
    script = quark_io_bench_bundle_preamble(cfg, "busybox")
    assert "/bench" in script
    assert "bind" in script
    assert "/var/lib/keska-lab/io-bench" in script


def test_concurrent_read_bench_uses_parallel_dd() -> None:
    sh = concurrent_read_bench_sh()
    assert "wait" in sh
    assert "CONCURRENT_READ_OK" in sh


def test_concurrent_read_mib_s() -> None:
    from keska_lab.harness.io_fs import concurrent_read_mib_s

    # 256 MiB in 1 second => 256 MiB/s
    assert concurrent_read_mib_s(total_bytes=268_435_456, elapsed_ns=1_000_000_000) == 256.0
