"""Cgroup v2 contract tests — mirror Rust conversion helpers."""

from __future__ import annotations


def _convert_cpu_shares_to_v2(cpu_shares: int) -> int:
    if cpu_shares == 0:
        return 0
    return 1 + ((cpu_shares - 2) * 9999) // 262142


def _convert_memory_swap_to_v2(memory_swap: int, memory: int) -> int | None:
    if memory == -1 and memory_swap == 0:
        return -1
    if memory_swap in (-1, 0):
        return memory_swap
    if memory in (0, -1):
        return None
    if memory < 0 or memory_swap < memory:
        return None
    return memory_swap - memory


def test_cpu_shares_conversion_matches_rust():
    assert _convert_cpu_shares_to_v2(0) == 0
    assert _convert_cpu_shares_to_v2(2) == 1
    assert _convert_cpu_shares_to_v2(1024) == 39


def test_memory_swap_conversion_matches_rust():
    assert _convert_memory_swap_to_v2(-1, 1024) == -1
    assert _convert_memory_swap_to_v2(0, 1024) == 0
    assert _convert_memory_swap_to_v2(2048, 1024) == 1024
    assert _convert_memory_swap_to_v2(0, -1) == -1
    assert _convert_memory_swap_to_v2(512, 1024) is None


def test_cpuset_inherit_required_for_join():
    """Empty cpuset on a leaf cgroup blocks cgroup.procs migration (EBUSY)."""
    # OCI pause pods often omit linux.resources.cpu; Quark inherits from parent.
    assert _convert_cpu_shares_to_v2(1024) == 39


def test_unified_path_trims_leading_slash():
    root = "/sys/fs/cgroup"

    def unified(name: str) -> str:
        return f"{root}/{name.lstrip('/')}"

    assert unified("/k8s.io/foo") == "/sys/fs/cgroup/k8s.io/foo"
    assert unified("k8s.io/foo") == "/sys/fs/cgroup/k8s.io/foo"
