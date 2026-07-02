# 036 — Sandboxed InitLoader Wait() ToSearch panic

**Tags:** upstream

## Symptom

CRI pause sandbox VM boots to `Shared-space ready` then guest panics on bootstrap vCPU:
`get panic : state is 0` in `vcpu_mgr.rs` (`ToSearch` assert). `crictl run` never completes.

## Cause

Bootstrap path in `rust_main` called `InitLoader()` synchronously before `WaitFn()`.
`LoadProcessKernel` → async `HostSpace::Call` → `taskMgr::Wait()` breaks vCPU scheduler
state (`ToSearch` panic or hang). Sync `HCall` from bootstrap also blocked progress.

## Fix

- Run `InitLoader` + `ControllerProcessHandler` from `BootstrapTask` after `WaitFn` starts (must not skip `InitLoader` — `LOADER.InitKernel` required for control socket handlers).
- Use sync `HCall` for `LoadProcessKernel` inside that task.
- `qlib/kernel/vcpu.rs` — `SwitchToRunning` ensures `Searching` before `ToRunning`

## Files

- `qkernel/src/lib.rs`
- `qlib/kernel/vcpu.rs`

## Verify

```bash
make -C qkernel TOOLCHAIN=nightly-2024-07-01 release
sudo cp build/qkernel.bin /usr/local/bin/qkernel.bin /usr/local/bin/qkernel_d.bin
sudo crictl run --no-pull container.json pod.json
```

Lab 2026-07-01: guest reaches `Shared-space ready` without panic after this fix; `crictl run` still hits ~4s containerd `DeadlineExceeded` on `StartRootContainer` (follow-up).
