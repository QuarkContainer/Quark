# 045 — ctr-pull devmapper digest fail after mirror docker import

## Symptom

Kata `ctr-pull` fails after mirror docker pull succeeds:

```
ctr: content digest sha256:c6348fa86ba0...: not found
```

## Cause

`ctr_pull_with_mirror_script` tried mirror/upstream `ctr pull` without GCP credentials, fell back to `docker pull` + `ctr import`, then re-pulled `docker.io/library/...` for devmapper unpack — wrong registry/digest vs the mirror image.

## Fix

- `ctr_pull_ref()` helper: `gcloud auth print-access-token` for Keska mirror refs on all `ctr pull` paths.
- Devmapper unpack uses the mirror ref that docker pulled (`dm_src`), not canonical docker.io.

## Files

- `lab/src/keska_lab/setup/image_registry.py`

## Verify

```python
lab.bench_all("light", n=10, setup=True)
```
