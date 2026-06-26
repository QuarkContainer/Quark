# 002 — ctr_image_ref double-tag on short image names

## Symptom

Kata `workloads` setup fails on `python:3.12-slim`: `ctr: failed to resolve reference "docker.io/library/python:3.12-slim:latest"` with HTTP 400 from Docker Hub.

## Cause

`ctr_image_ref()` treated any image without `/` as untagged and appended `:latest`, even when a tag was already present (`python:3.12-slim` → `python:3.12-slim:latest`).

## Fix

If `:` is present in a short name, use `docker.io/library/{image}` as-is. Also allow `CtrImagePullStep` to fall through to docker-save import when devmapper pull fails.

## Files

- `lab/src/keska_lab/setup/oci_bundle.py`

## Verify

```bash
cd lab && PYTHONPATH=src python3 -c "from keska_lab.setup.oci_bundle import ctr_image_ref; assert ctr_image_ref('python:3.12-slim') == 'docker.io/library/python:3.12-slim'"
lab.kata.bench("workloads", workload="python", n=1, setup=True)
```
