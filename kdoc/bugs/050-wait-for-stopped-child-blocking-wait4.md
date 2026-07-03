# WaitForStopped child path uses blocking wait4

## Symptom

When the sandbox process is a direct child (`child == true`), `WaitForStopped()` polled with `wait4(WNOHANG)` in a 10ms loop instead of reaping the zombie in one call.

## Cause

gVisor-style child reap was not adopted: after `Destroy()` sends SIGKILL, a blocking `wait4(0)` is the correct primitive. WNOHANG polling added latency and could miss a clean reap race.

## Fix

- **`child == true`:** blocking `wait4(pid, &status, 0)`; clear `Pid` on success; treat `ECHILD` as already reaped.
- **`child == false`:** unchanged — 500ms `IsRunning()` poll, force SIGKILL, clear `Pid`, return `Ok(())` (see [048](048-quark-destroy-stale-metadata-list-hang.md)).

## Files

- `qvisor/src/runc/sandbox/sandbox.rs`

## Verify

```bash
# After deploy: create/start/delete in same quark process reaps sandbox child without zombie leak
ps aux | grep '[q]visor boot'  # none after delete
```
