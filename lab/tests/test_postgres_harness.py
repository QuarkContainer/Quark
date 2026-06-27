"""Tests for postgres harness helpers."""

from keska_lab.harness.postgres import oci_mount_to_ctr
from keska_lab.harness.workload import get_workload


def test_oci_mount_to_ctr_postgres_shm() -> None:
    spec = get_workload("postgres")
    shm = spec.oci_mounts[0]
    flag = oci_mount_to_ctr(shm)
    assert flag.startswith("type=tmpfs,")
    assert "dst=/dev/shm" in flag
    assert "size=64m" in flag
