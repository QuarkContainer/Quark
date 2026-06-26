# 003 — devmapper base_image_size too small for python layers

## Symptom

Kata `ctr-pull` / smoke for `python:3.12-slim` fails: `no space left on device` during layer extract to devmapper, while `busybox` works.

## Cause

containerd devmapper `base_image_size` was `16MB`; python image layers exceed thin-device capacity during unpack.

## Fix

Raise `base_image_size` to `512MB`, default devmapper data pool to `50G`, and patch existing lab config on setup. Do not treat docker-import fallback as success when Kata uses devmapper snapshotter.

## Files

- `lab/src/keska_lab/setup/kata_firecracker.py`
- `lab/src/keska_lab/setup/oci_bundle.py`

## Verify

```bash
sudo ctr images pull --snapshotter devmapper --local docker.io/library/python:3.12-slim
lab.kata.bench("workloads", workload="python", n=1, setup=True)
```
