# Quark force-delete fallback used literal `ID` path

## Symptom

Light suite batch hung during `memory_idle_rss_mb` after a few samples. `quark list` on the lab host could block; orphaned qvisor processes and stale metadata under `/run/qvisor/` and `/var/lib/quark/` accumulated.

## Cause

`_quark_force_delete` parsed `"$ID"` into the Python string `ID` and emitted fallback cleanup as `/run/qvisor/ID` and `awk -v id="ID"`. When the 5s `quark delete` timed out, kill/rm_meta never targeted the real sandbox, leaving ghosts that eventually stalled create/list.

Lowering delete timeout from 20s to 5s made the broken fallback trigger more often.

## Fix

Use the shell id token at runtime: `rm -rf "/run/qvisor/$ID"`. Read sandbox PID from `/run/qvisor/$ID/meta.json` (not `quark list`) when delete times out. Wrap batch `create`/`start` with `timeout`. Add unit tests for generated snippet.

See also [048](048-quark-destroy-stale-metadata-list-hang.md) for runtime metadata removal and `WaitForStopped` timeout.

## Files

- `lab/src/keska_lab/backends/quark.py`
- `lab/tests/test_harness_batch.py`

## Verify

```bash
cd lab && python -m pytest tests/test_harness_batch.py::test_quark_force_delete_uses_runtime_id -q
lab.cleanup()
lab.quark.bench("light", n=10, setup=False)  # completes, streams METRIC lines
```
