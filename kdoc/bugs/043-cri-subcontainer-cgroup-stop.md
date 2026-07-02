# 043 — CRI subcontainer cgroup + stop

## Symptom

- `crictl stats` on workload containers: empty memory / RPC ok but no metrics (H2).
- Multi-container pod: second container still running after `crictl stop` (H3).
- `state` RPC always reported pid 123.

## Cause

- Subcontainer `Sandbox` stub copied only `ID`/`Pid`, not pod `Cgroup`.
- Global `SANDBOX` lock never stored `Cgroup` from pause sandbox create.
- `Container::Stop()` cleared `Sandbox` for subcontainers before `WaitforStopped`, skipping wait.
- `shim_task::state` overwrote real pid with 123.

## Fix

- Propagate `Cgroup` to `SANDBOX` on first shim Task create; clone into subcontainer stub.
- Only clear `Sandbox` on root container stop; honor `Status::Stopped` in `WaitforStopped`.
- Remove hardcoded pid override in `state`.
- `MetricsFromCgroup`: cgroup v2 unified paths (`memory.current` under `/sys/fs/cgroup/k8s.io/...`).
- Shim `stats`: prefer container spec `cgroupsPath` over sandbox stub.

## Files

- `qvisor/src/runc/container/container.rs`
- `qvisor/src/runc/shim/shim_task.rs`, `shim/container.rs`
- `qvisor/src/runc/cgroup/cgroup.rs`, `cgroup/stats.rs`
- `lab/src/keska_lab/cri/api_spot.py` (L3: stats RPC; Kata JSON memory; optional host cgroup)

## Verify

```bash
keska-lab-cri-gate --runtime quark --layer L5
keska-lab-cri-bisect --bug 043   # optional isolated proof
```
