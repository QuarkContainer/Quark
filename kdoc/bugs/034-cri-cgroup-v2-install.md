# 034 — CRI cgroup install used v1 paths on cgroup v2 host

**Tags:** upstream, env

## Symptom

Quark CRI `crictl run` fails with `WriteFile .../cpu/k8s.io/.../cpu.shares Permission denied` on lab (cgroup v2 unified).

## Cause

`Cgroup::Install` only checked v1 `memory` controller paths and created v1 controller dirs. Lab host uses cgroup v2; `cpu.shares` does not exist.

## Fix

Detect unified cgroup v2 (`/sys/fs/cgroup/cgroup.controllers`). Use v2 install/join/uninstall (`cpu.weight`, `memory.max`, etc.) via existing `cgroup_v2` controllers.

## Files

- `qvisor/src/runc/cgroup/cgroup.rs`

## Verify

```bash
sudo crictl run container.json pod.json
keska-lab-network-preflight
```
