# 038 — CRI sandbox double PivotRoot chdir fail

**Tags:** upstream, cri

## Symptom

After VcpuWait fix, `crictl runp` returns quickly but sandbox VM panics:

```
chdir fail for rootfs /var/lib/quark/<sandbox-id>
```

`/var/lib/quark/<id>` exists before pivot; panic is on the second `LoadProcessKernel`.

## Cause

Sandboxed boot with `autoStart` creates two guest tasks that both call host
`LoadProcessKernel`, which always ran `PivotRoot`:

1. `BootstrapTask` → `InitLoader` → `LoadProcessKernel` → pivot (ok)
2. `StartRootContainer` → `LoadProcessKernel` → `chdir` to absolute
   `/var/lib/quark/<id>` fails (already inside pivoted root)

## Fix

- `qvisor/src/vmspace/mod.rs` — track `pivoted` on `VMSpace`; `pivotOnce()`
  skips repeat `PivotRoot` in `LoadProcessKernel`.
- `qvisor/src/runc/runtime/sandbox_process.rs` — task-service path uses
  `pivotOnce` as well.

## Files

- `qvisor/src/vmspace/mod.rs`
- `qvisor/src/runc/runtime/sandbox_process.rs`

## Verify

```bash
sudo crictl runp pod.json
sudo crictl run --no-pull container.json pod.json
sudo crictl exec <cid> echo hello
```
