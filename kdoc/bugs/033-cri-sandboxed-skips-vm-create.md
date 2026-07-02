# 033 — CRI Sandboxed mode skipped pod sandbox VM create

**Tags:** upstream

## Symptom

Quark CRI with `Sandboxed=true`: `crictl run` fails with `UCliSocket connect socket fail, path is qvisor-sandbox., error is 111`.

## Cause

`Container::Create` always called `CreateSubContainer` when `Sandboxed` was set, even for the first (pause/pod sandbox) container. No VM or control socket was started.

## Fix

- Create pod sandbox VM when `Sandboxed` and global `SANDBOX.ID` is still empty (same path as `IsRoot`).
- Only use `CreateSubContainer` when `Sandboxed` and parent sandbox already exists.
- Always populate `SANDBOX` global on first shim container.

## Files

- `qvisor/src/runc/container/container.rs`
- `qvisor/src/runc/shim/shim_task.rs`

## Verify

```bash
sudo crictl run container.json pod.json   # busybox smoke
keska-lab-network-preflight
```
