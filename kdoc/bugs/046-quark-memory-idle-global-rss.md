# Bug 046 — Quark memory_idle sums global RSS

## Symptom

`memory_idle_rss_mb` and `memory_while_paused_rss_mb` for Quark in the light suite reported ~65 MB vs a historical ~21 MB baseline, while `tti_ms` stayed ~55 ms (no runtime regression).

## Cause

`memory_idle_once` / `pause_resume_once` summed RSS for **all** host processes matching `quark|qvisor|qemu|…` globally. Stale CRI pods from lab gates (`keska-*`) were not torn down by cleanup, inflating the total. Kata already filtered by sandbox ID in process args.

## Fix

- Per-sandbox RSS via `quark list` PID + shallow process tree, with args-match fallback (`harness/metrics.py`).
- CRI-aware cleanup: `crictl` pods named `keska*`, tmpfs/io workdirs (`setup/quark_cleanup.py`).
- Pre-memory `backend.cleanup()` in harness runner.

## Files

- `lab/src/keska_lab/harness/metrics.py`
- `lab/src/keska_lab/backends/quark.py`
- `lab/src/keska_lab/setup/quark_cleanup.py`
- `lab/src/keska_lab/harness/runner.py`

## Verify

```python
lab.bench_all("light", n=3, setup=True)
# Quark memory_idle_rss_mb ~20–25 MB on a clean lab; Kata unchanged (~160 MB)
```
