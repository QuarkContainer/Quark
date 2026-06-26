# 008 — docker-ensure false fail + kata devmapper setup hang

## Symptom

- `lab.bench_all("db", setup=True)` fails at `docker-ensure: command failed (no output)`.
- Kata `kata-firecracker` / devmapper health hangs for minutes with no output; many orphaned `firecracker` processes on lab.

## Cause

1. Multiline setup scripts over SSH (`bash -lc`, embedded `exit 0`) returned wrong exit codes and swallowed stderr (`docker info >/dev/null`).
2. `cleanup-sandboxes` only stopped Quark VMs — stale `keska-kata-*` ctr containers blocked devmapper health `ctr run`.
3. Devmapper health had no timeout; unhealthy pool with existing `devpool` skipped recovery.

## Fix

- Run multiline scripts via SSH stdin temp file (`remote._run_script`); remove early `exit 0` from embedded docker ensure.
- Extend cleanup: `killall firecracker` first, then kill/remove `keska-*` ctr containers (with per-op timeout).
- Devmapper: `timeout 45` on health `ctr run`, `clear_stale_kata()` when pool exists but unhealthy, better docker-ensure error output.
- Quark postgres: tmpfs mount at `/var/run/postgresql` (uid 70 cannot use rootfs dir from docker export).

## Files

- `lab/src/keska_lab/remote.py`
- `lab/src/keska_lab/setup/docker.py`
- `lab/src/keska_lab/setup/quark_cleanup.py`
- `lab/src/keska_lab/setup/kata_firecracker.py`
- `lab/src/keska_lab/harness/workload.py`

## Verify

```bash
cd lab && python3 -c "from keska_lab.session import LabSession; lab=LabSession(); lab.bench_all('db', n=2, setup=True)"
```

Both backends: setup ok, `pgbench_tps` > 0, zero errors.
