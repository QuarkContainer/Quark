# Quark Benchmark Suite

Wave-based performance and resource benchmarks for Quark on the Lima dev VM.
Reports are JSON for comparing runs over time (security, optimization, utilization).

## Quick start

```bash
python3 quark.py bench setup          # once: install fio, iperf3 in VM
python3 quark.py bench cleanup        # if prior runs left orphaned quark_d
python3 quark.py bench run            # dev profile (all suites)
python3 quark.py bench run --profile stress
python3 quark.py bench results
python3 quark.py bench compare benchmark/results/run_a.json benchmark/results/run_b.json
```

Or directly:

```bash
python3 benchmark/bench.py run --profile dev --suite cold_start
python3 benchmark/compare.py benchmark/results/*.json  # needs two files
```

## Profiles

| Profile | Wave size | Waves | ~Cold starts |
|---------|-----------|-------|--------------|
| `dev` | 20 | 5 | 100 |
| `stress` | 50 | 20 | 1000 |

Override: `--wave-size 30 --waves 10`

## Suites

| Suite | Metrics |
|-------|---------|
| `cold_start` | Container start latency (ms), per-wave stats |
| `hibernate` | pause/resume latency, RSS before/after `docker pause` |
| `memory` | Idle per-container MB, loaded RSS |
| `network_cluster` | iperf3 TCP between two alpine containers |
| `network_inet` | Real TCP connect time + HTTPS download |
| `io_fs` | busybox dd read/write 64 MiB in container |

Run one suite: `--suite hibernate`

## Output

JSON written to `benchmark/results/<timestamp>_<git-sha>.json`:

- `metadata` — git commit, runtime, profile, lima info
- `suites` — per-suite stats (mean, p50, p95, p99)
- `host_snapshots` — MemAvailable, load average

## Compare over time

```bash
python3 benchmark/compare.py baseline.json after_my_change.json
python3 benchmark/compare.py a.json b.json --json-delta delta.json
```

Regression flag when a metric worsens by >10% (configurable in `compare.py`).

## Variables

| Env / flag | Default | Purpose |
|------------|---------|---------|
| `RUNTIME` / `--runtime` | `quark_d` | Docker runtime |
| `--profile` | `dev` | Wave sizing |
| `--output` | auto | Result path |

## Notes

- All work runs **inside Lima** via `scripts/dev/bin/vm-exec.sh` — not Docker Desktop.
- Aborts if `MemAvailable` drops below ~500 MiB mid-run.
- Internet tests record `skipped` when offline.
- Hibernate uses `docker pause` → Quark memory swap-out path.

## Troubleshooting slow or failing cold_start

**Symptoms:** `warning: N containers failed in wave`, multi-second or 20+ second latencies, veth errors.

**Common causes:**

1. **Orphaned `quark_d` processes** from prior benchmark runs (load avg >> CPU count). Each leaked process consumes CPU and slows every new start.
   ```bash
   python3 quark.py bench cleanup
   ```

2. **Parallel veth races** — Docker can fail with `error renaming interface veth… to eth0: file exists` when many Quark containers start at once. The benchmark uses 50 ms stagger by default; increase with `--stagger-ms 100` or reduce `--wave-size`.

3. **Wrong host** — never run `docker run --runtime=quark_d` on macOS Docker Desktop. Use `quark.py` or `limactl shell quark`.

**Sanity check** (sequential, perf_test-style):
```bash
python3 quark.py run compare   # runc vs quark_d, ~1–2 s expected on Lima aarch64
```

**Fair cold_start** (after cleanup):
```bash
python3 benchmark/bench.py cleanup
python3 benchmark/bench.py run --suite cold_start --wave-size 5 --waves 3 --image busybox -v
```

See also: [scripts/dev/README.md](../scripts/dev/README.md)
