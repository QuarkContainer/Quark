"""Per-bug CRI fix proof — revert one bug's files to baseline, gate must fail; restore must pass."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
BASELINE = "a353776f"
FIX_BACKUP = REPO_ROOT / ".keska-cri-fix-backup"


@dataclass(frozen=True)
class BugProof:
    bug_id: str
    files: tuple[str, ...]
    layer: str
    symptom: str
    rebuild_qkernel: bool = False


# Ordered apply chain (stash-index). Files may overlap — negative test reverts whole files.
BUG_CHAIN: tuple[BugProof, ...] = (
    BugProof(
        "032",
        ("qvisor/src/main.rs",),
        "L0",
        "quark CLI enters shim when argv0 is quark (fixed: argv0 containerd-shim-* only)",
    ),
    BugProof(
        "031",
        ("qvisor/src/runc/shim/container.rs",),
        "L2",
        "bundle path /{id}/options.json ENOENT under containerd 2.x",
    ),
    BugProof(
        "033",
        (
            "qvisor/src/runc/container/container.rs",
            "qvisor/src/runc/shim/shim_task.rs",
        ),
        "L2",
        "Sandboxed pod skips VM create on shim Task create",
    ),
    BugProof(
        "034",
        ("qvisor/src/runc/cgroup/cgroup.rs",),
        "L2",
        "cgroup v2 cpu.shares permission denied on install",
    ),
    BugProof(
        "035",
        (
            "qvisor/src/runc/runtime/vm_type/noncc.rs",
            "qvisor/src/runc/runtime/vm_type/emulcc.rs",
            "qvisor/src/runc/runtime/vm_type/sevsnp.rs",
            "qkernel/src/lib.rs",
        ),
        "L2",
        "pause sandbox allocates all host vCPUs / boot hang",
        rebuild_qkernel=True,
    ),
    BugProof(
        "036",
        (
            "qkernel/src/lib.rs",
            "qlib/kernel/vcpu.rs",
            "qlib/kernel/taskMgr.rs",
        ),
        "L2",
        "InitLoader Wait ToSearch panic (state is 0)",
        rebuild_qkernel=True,
    ),
    BugProof(
        "037",
        (
            "qvisor/src/kvm_vcpu.rs",
            "qlib/kernel/taskMgr.rs",
            "qkernel/src/lib.rs",
            "qvisor/src/runc/runtime/sandbox_process.rs",
        ),
        "L2",
        "single-vCPU io_uring Accept deadlock (~120s timeout)",
        rebuild_qkernel=True,
    ),
    BugProof(
        "038",
        (
            "qvisor/src/vmspace/mod.rs",
            "qvisor/src/runc/runtime/sandbox_process.rs",
        ),
        "L2",
        "second LoadProcessKernel chdir fail (double pivot)",
    ),
    BugProof(
        "039",
        ("qvisor/src/runc/sandbox/sandbox.rs",),
        "L5",
        "subcontainer rootfs path wrong in multi-container pod",
    ),
    BugProof(
        "043",
        (
            "qvisor/src/runc/container/container.rs",
            "qvisor/src/runc/shim/shim_task.rs",
            "qvisor/src/runc/shim/container.rs",
            "qvisor/src/runc/cgroup/cgroup.rs",
            "qvisor/src/runc/cgroup/stats.rs",
        ),
        "L5",
        "subcontainer stop race / cgroup propagation / stats path",
    ),
)


def all_cri_files() -> frozenset[str]:
    out: set[str] = set()
    for b in BUG_CHAIN:
        out.update(b.files)
    return frozenset(out)


def backup_fixed_tree() -> None:
    """Snapshot fixed files from working tree before any revert."""
    FIX_BACKUP.mkdir(parents=True, exist_ok=True)
    for rel in all_cri_files():
        src = REPO_ROOT / rel
        if not src.is_file():
            continue
        dst = FIX_BACKUP / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def restore_fixed_files(files: tuple[str, ...]) -> None:
    for rel in files:
        src = FIX_BACKUP / rel
        dst = REPO_ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(f"missing backup {src}; run backup first")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def touch_rebuild_hints(files: tuple[str, ...]) -> None:
    """Bump mtimes so lab provision rebuilds qvisor/qkernel after restore."""
    for rel in (*files, "qlib/config.rs", "qkernel/src/lib.rs"):
        p = REPO_ROOT / rel
        if p.is_file():
            p.touch()


def revert_files_to_baseline(files: tuple[str, ...]) -> None:
    subprocess.run(
        ["git", "checkout", BASELINE, "--", *files],
        cwd=REPO_ROOT,
        check=True,
    )


def check_032_local() -> tuple[bool, str]:
    """032: main.rs must gate shim on argv0, not config."""
    path = REPO_ROOT / "qvisor/src/main.rs"
    text = path.read_text()
    if "invoked_as_containerd_shim()" in text and "ShimMode" not in text:
        return True, "argv0 shim entry present; ShimMode removed"
    return False, "main.rs missing invoked_as_containerd_shim or still references ShimMode"


def cumulative_files_through(bug_id: str) -> tuple[str, ...]:
    out: list[str] = []
    for b in BUG_CHAIN:
        out.extend(b.files)
        if b.bug_id == bug_id:
            break
    else:
        raise KeyError(bug_id)
    return tuple(dict.fromkeys(out))
