# 006 — quark exec stops postgres daemon sandbox

## Symptom

`pgbench_tps` returns exit 255 / empty output; `memory_idle_rss_mb` inconsistent. Sandbox `state` flips to `stopped` after `quark exec ... pg_isready`.

## Cause

On Quark direct OCI, `exec` into a postgres daemon sandbox tears down the VM after the exec completes. A readiness loop calling `pg_isready` before `pgbench` leaves nothing running for the benchmark exec.

## Fix

- **TTI:** keep `pg_isready` exec (probe is the metric; sandbox may stop after).
- **memory_idle / pgbench:** wait with `sleep 3` after `start`, no pre-exec probes; run `pgbench` as the first (and only) exec.

## Files

- `lab/src/keska_lab/backends/quark.py`

## Verify

```python
lab.quark.bench("db", n=2, setup=True)  # pgbench_tps > 0
```
