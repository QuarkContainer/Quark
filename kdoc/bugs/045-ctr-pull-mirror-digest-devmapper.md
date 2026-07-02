# 045 — ctr-pull devmapper digest fail after mirror docker import

## Symptom

Kata `ctr-pull` fails after mirror docker pull succeeds:

```
ctr: content digest sha256:c6348fa86ba0...: not found
```

## Cause

1. **containerd 2.x `ctr`** auth is `-u oauth2accesstoken:TOKEN`, not `--user … --secret` (containerd 1.x).
2. **Devmapper unpack** needs a `transfer.v1.local` `unpack_config` for `devmapper`; kata setup skipped it because `snapshotter = 'devmapper'` already appeared on the kata CRI runtime block.
3. Without mirror `ctr` auth + devmapper unpack, the script fell back to `docker pull` + re-pull `docker.io/library/...` for devmapper — wrong digest.

## Fix

- `ctr_pull_ref()`: `-u "oauth2accesstoken:$token"` for Keska mirror pulls; devmapper unpack uses mirror ref.
- `containerd_cri.py` / `kata_firecracker.py`: add devmapper `unpack_config` stanza (regex check, not global `snapshotter = 'devmapper'`).

## Files

- `lab/src/keska_lab/setup/image_registry.py`
- `lab/src/keska_lab/setup/containerd_cri.py`
- `lab/src/keska_lab/setup/kata_firecracker.py`

## Verify

```python
lab.bench_all("light", n=10, setup=True)
```
