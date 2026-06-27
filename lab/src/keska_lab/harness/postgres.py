"""Shared postgres benchmark helpers (Quark OCI ↔ Kata ctr mounts)."""

from __future__ import annotations

from keska_lab.harness.workload import get_workload


def oci_mount_to_ctr(m: dict) -> str:
    """Convert an OCI mount dict to a containerd --mount flag value."""
    opts = m.get("options") or []
    opt_str = ":".join(opts) if opts else "rw"
    return f"type={m['type']},src={m['source']},dst={m['destination']},options={opt_str}"


def kata_postgres_ctr_mounts_shell() -> str:
    """Shell assignments for PGDATA + workload tmpfs mounts (matches Quark OCI bundle)."""
    spec = get_workload("postgres")
    shm = oci_mount_to_ctr(spec.oci_mounts[0])
    pg_run = oci_mount_to_ctr(spec.oci_mounts[1])
    return (
        f'PG_SHM_MOUNT="{shm}"\n'
        f'PG_RUN_MOUNT="{pg_run}"'
    )


POSTGRES_STARTUP_SLEEP_SECS = 3
