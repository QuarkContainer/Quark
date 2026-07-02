# 044 — cgroup v1 support removed; v2-only path

**Tags:** upstream, cri, cgroup

## Symptom

Dual cgroup v1/v2 code in `cgroup.rs` was hard to review, untested on v1, and duplicated logic already in `cgroup_v2.rs`.

## Cause

Bug 034 bolted v2 onto gVisor-era v1 `CONTROLLERS` loop (~400 lines). Lab and target K8s nodes use unified cgroup v2 only.

## Fix

- `cgroup.rs`: v2-only `Install` / `Join` / `Uninstall`; `create_unified_hierarchy` with `cgroup.subtree_control` delegation; `require_cgroup_v2()`.
- `stats.rs`: read `memory.current` and `cpu.stat` from unified path only.
- Removed v1 controllers (`cpu.shares`, per-controller hierarchies, `MakePath`, `LoadPaths`, blkio/net_cls).
- Fixed `count_cpuset` range parsing (was using wrong index for end).
- `CpuSet2` inherits `cpuset.cpus` / `cpuset.mems` from ancestors when OCI spec omits them (required for containerd pre-created pod cgroups; empty cpuset causes `EBUSY` on join).
- CRI pod cgroups have `cgroup.subtree_control` set; processes join a leaf at `{path}/init` instead of the delegation parent.
- `create_unified_hierarchy` no longer enables `subtree_control` on the leaf cgroup.
- Tests: unit tests in `cgroup.rs` + `cgroup_v2.rs`; Linux integration `cgroup_v2_install_join_roundtrip`.

## Files

- `qvisor/src/runc/cgroup/cgroup.rs`
- `qvisor/src/runc/cgroup/cgroup_v2.rs`
- `qvisor/src/runc/cgroup/stats.rs`
- `lab/tests/test_cgroup_v2.py`

## Verify

```bash
cd qvisor && cargo test cgroup
keska-lab-cri-gate --runtime quark --layer L5
pytest lab/tests/test_cgroup_v2.py
```

**Requirement:** host must mount unified cgroup v2 at `/sys/fs/cgroup`.
