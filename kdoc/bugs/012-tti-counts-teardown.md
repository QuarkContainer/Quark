# 012 — TTI metric included sandbox teardown

## Symptom

Quark busybox TTI ~120 ms vs historic ~62 ms; postgres TTI ~3090 ms vs ~60 ms. Looked like kernel/regression; pgbench TPS unchanged.

## Cause

- **Busybox:** `_direct_lifecycle_script` stopped the timer after `_quark_force_delete()` (~60 ms teardown in the metric).
- **Postgres:** `_postgres_startup_sleep()` (`sleep 3`) was wired into the TTI script; bug 006 intended that wait only for pgbench/memory_idle.

## Fix

- Stop TTI clock immediately after the readiness exec; run cleanup afterward.
- Remove `sleep 3` from postgres TTI (keep on pgbench/memory_idle paths).

## Files

- `lab/src/keska_lab/backends/quark.py`

## Verify

```python
lab.quark.bench("tti", n=5)
lab.quark.bench("db", n=2, setup=True)  # tti_ms ~60, pgbench_tps unchanged
```
