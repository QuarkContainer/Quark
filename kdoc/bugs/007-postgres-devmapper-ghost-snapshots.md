# 007 — postgres devmapper unpack: ghost snapshot metadata

## Symptom

Kata `ctr-pull` for `postgres:16-alpine` fails on devmapper unpack: `failed to prepare extraction snapshot ... sha256:3501ee8b...: snapshot does not exist`. `busybox` and `python:3.12-slim` work.

## Cause

Partial postgres unpacks left snapshot keys in containerd `meta.db` (often in `k8s.io`) while devmapper snapshotter state was reset or never committed. Unpack then expects chain snapshots like `34884…` / `3501…` that exist in metadata but not in the devmapper plugin.

## Fix

Before devmapper pulls: `ctr images rm --sync` across `default`/`k8s.io`/`moby`, leaf-delete orphaned snapshots, and on snapshot errors reset devmapper `metadata.db` after sync rm. Fall back to `docker save | ctr import` then devmapper unpack. Use `--sync` in devmapper pool reset image cleanup too.

## Files

- `lab/src/keska_lab/setup/oci_bundle.py`
- `lab/src/keska_lab/setup/kata_firecracker.py`

## Verify

```python
lab.kata.bench("db", n=2, setup=True)
```
