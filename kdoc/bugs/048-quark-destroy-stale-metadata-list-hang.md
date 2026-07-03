# Quark destroy leaves metadata; list hangs under churn

## Symptom

Light suite batch hung during `tti_under_load` or `memory_idle_rss_mb`. `timeout 3 sudo quark list` blocked on the lab host. Stale dirs under `/run/qvisor/` accumulated; orphaned `qvisor boot` at 100% CPU made deletes and list worse.

## Cause

1. `Container::Destroy()` stopped the sandbox and wrote `Status: Stopped` but never removed `/run/qvisor/<id>/`.
2. `quark list` does O(n) `Container::Load` with exclusive flock per entry — slow with stale metadata.
3. `Sandbox::WaitForStopped()` waited 5s then returned `Err`, aborting cleanup when a VM was stuck.
4. Harness `_quark_force_delete` fallback called `quark list` for PID lookup, blocking when list was slow.

Related: [047](047-quark-force-delete-wrong-id-path.md), [015](015-lab-sandbox-orphan-leak.md).

## Fix

- **Runtime:** `Destroy()` best-effort `fs::remove_dir_all(&self.Root)` after stop; append rm errors to existing `errs` slice.
- **Runtime:** `WaitForStopped()` deadline 500ms; on expiry second SIGKILL, brief wait4, clear `Pid`, return `Ok(())`.
- **Harness:** read sandbox PID from `/run/qvisor/$ID/meta.json` via `python3`, no `quark list` in fallback.

## Files

- `qvisor/src/runc/container/container.rs`
- `qvisor/src/runc/sandbox/sandbox.rs`
- `lab/src/keska_lab/backends/quark.py`
- `lab/tests/test_harness_batch.py`

## Verify

```bash
cd lab && python -m pytest tests/test_harness_batch.py -k force_delete -q
# After deploy: 20 create/start/delete cycles leave zero keska-* under /run/qvisor
lab.quark.bench("light", n=10, setup=False)  # completes in <45s
```
