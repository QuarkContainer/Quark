# 004 — Postgres OCI bundle + pgbench on Quark

## Symptom

`db` suite on Quark: postgres sandbox exits immediately, `pgbench -i` stops the VM, or pgbench benchmark returns empty/255.

## Cause

1. Shared `PGDATA` in the OCI bundle rootfs corrupts across concurrent sandboxes.
2. `docker-entrypoint.sh` as root cannot `chown` data dirs in the microVM.
3. `pgbench -i` inside a running Quark sandbox can crash postgres (sandbox goes `stopped`).
4. Alpine `pgbench` rejects `--progress=none`.
5. `/var/run/postgresql` from docker export is not writable by uid 70 inside the microVM.

## Fix

- Pre-init cluster + `pgbench -i` into `postgres-data-template` via Docker (`docker exec --user postgres`).
- Per-run bundle: symlink `rootfs`, bind-mount isolated `data/` dir, run as uid 70.
- Benchmark path runs `pgbench -c1 -T5` only (no `-i` in hot path).
- Add tmpfs mount at `/var/run/postgresql` (mode 1777) so postgres can create socket lock files.
- Lab bench: PGDATA bind mount backed by host tmpfs; `UringIO` in `/etc/quark/config.json`.

## Verify

```bash
lab.bench_all("db", n=2, setup=True)  # quark pgbench_tps ~300+ (host bind still below kata devmapper)
```
