# 015 — Lab sandbox orphan leak under load

## Symptom

Group 5 E1 experimental runs hung for 45+ min with no bench output after deploy. Lab accumulated many `quark boot` processes at ~17 MB RSS and 100% CPU; each `tti_under_load` iteration took ~5 min instead of seconds.

## Cause

1. `tti_under_load_once` used unbounded `quark delete --force` (no timeout). When delete hung, the SSH script hit its 360s timeout and was killed without running the EXIT trap — sandboxes were left running.
2. The script also `rm -rf`’d `/run/qvisor/keska-*` at the start of each iteration without killing VMs, so metadata was wiped while orphaned `quark boot` processes kept spinning.
3. `_quark_force_delete` fallback used `pkill -9 quark boot`, which is unsafe with multiple concurrent sandboxes.

## Fix

- `_quark_force_delete`: timeout delete, then kill the sandbox PID from `quark list`, then remove stale metadata — per sandbox, not global pkill.
- Use force-delete in `tti_under_load`, pause/resume, io benches, and python exec paths.
- Remove destructive `rm -rf keska-*` preamble from `tti_under_load`.
- Harness: cleanup after `tti_under_load` iterations and on any bench exception; group5 cleans up before each suite.

## Files

- `lab/src/keska_lab/backends/quark.py`
- `lab/src/keska_lab/harness/runner.py`
- `lab/src/keska_lab/group5.py`

## Verify

```bash
KESKA_LAB_SKIP_REGISTRY_AUTH=1 keska-lab-group5 UringFixedFile -n 5 --experimental-only
# light suite completes in minutes; no accumulating quark boot processes on lab
ssh lab@lab.keska.vpn 'ps aux | grep "[q]uark boot" | wc -l'  # expect 0–1 during run
```
