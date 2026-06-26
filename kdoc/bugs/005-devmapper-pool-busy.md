# 005 — devmapper pool create: Device or resource busy

## Symptom

Kata setup fails at `kata-firecracker`: `device-mapper: create ioctl on devpool failed: Device or resource busy`.

## Cause

Stale thin volumes or Kata containers still hold the devmapper pool after a failed/interrupted run. `dmsetup remove devpool` alone fails silently, then recreate hits EBUSY.

## Fix

`force_remove_pool()` in `devmapper_pool_script`: stop containerd, kill/remove ctr tasks and images, remove child thin devices (newest first), detach losetup, retry create on failure. Full reset if health check still fails.

## Files

- `lab/src/keska_lab/setup/kata_firecracker.py`

## Verify

```python
lab.kata.bench("db", n=1, setup=True)
```
