# Keska Lab

IPython-native laboratory on **`lab@lab.keska.vpn`** — experiment with **Quark** and **Kata Containers**, build/deploy Quark from your Mac, and run benchmarks with automatic setup.

## Install (Mac or Linux)

```bash
cd lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
ssh lab@lab.keska.vpn   # verify key-based auth
keska-lab
```

## Architecture

```
Your machine (keska-lab / IPython)
    │  SSH
    ▼
lab.keska.vpn (x86_64, KVM)
    ├── QuarkEnvironment  →  direct OCI: quark create/start/exec/delete
    └── KataEnvironment   →  containerd ctr --runtime io.containerd.kata.v2
```

**Hot path (timed):** no dockerd. Setup may use Docker once to export a cached OCI bundle under `/tmp/keska-lab/bundles/`.

Default Quark binary is **release `quark`**, not debug `quark_d`. Kata default hypervisor is **Firecracker** (devmapper snapshotter).

---

## Quark: build & deploy

```python
lab.quark.run()
lab.quark.bench("light", n=10)
lab.quark.bench("workloads", workload="python", n=5)
lab.quark.bench("network", n=3, setup=True)   # CRI stack; TSOT when KESKA_LAB_ENABLE_TSOT=1
```

Network suite setup (automatic with `setup=True`): CNI bridge plugins, containerd CRI with Quark/Kata shim, crictl, and iptables forward rules for `cni0`. With `KESKA_LAB_ENABLE_TSOT=1`, Quark setup also deploys the TSOT stack (etcd, cadvisor, qservice `na`) and network probes run via crictl with sandbox UIDs.

---

## Kata Containers

```python
lab.kata.run()
lab.kata.bench("light", n=10)
```

Firecracker workloads beyond `busybox` need devmapper image space on the lab host.

---

## Benchmark harness

Composable suites via `keska_lab.harness`:

| Suite | Contents | Backends |
|-------|----------|----------|
| `light` | TTI, TTI under load, memory, pause/resume | Quark + Kata |
| `full` | `light` + host-backed `io_write` / `io_read` / `io_concurrent_read` | Quark + Kata |
| `workloads` | TTI + memory idle per workload | Quark + Kata |
| `standard` | `light` on busybox + `workloads` on python | Quark + Kata |
| `network` | inet connect/download, sandbox iperf (CRI) | Quark + Kata |
| `db` | postgres TTI, idle RSS, pgbench TPS | Quark + Kata |
| `heavy` | `full` + network (legacy) | Mixed |

### Measurement philosophy (Quark vs Firecracker/Kata)

**Firecracker via Kata is the performance reference.** Quark is expected to win on many metrics when the harness is fair — large gaps should reflect the runtimes, not how we measure them.

Rules the harness follows:

1. **Same lifecycle shape** — both backends: provision → start → exec probe/workload → **stop clock** → teardown. Never count delete/kill/rm in timed metrics (TTI, concurrent I/O wall time).
2. **Same guest work** — identical probe commands, postgres warm PGDATA template, host `/bench` bind mount for I/O, parallel reader count for concurrent read.
3. **Prep outside the timer** — bundle refresh and PGDATA copy happen before `t0` (Quark); Kata pulls images and ensures the same template during setup.
4. **Interpret ratios, not absolutes** — Kata busybox TTI ~500 ms vs Quark ~50 ms is a real stack gap with comparable methodology. Kata postgres TTI ~16 s vs Quark ~56 ms is likewise measured the same way (warm data, `pg_isready` loop); Quark is faster because the runtime is faster, not because Kata still counts teardown.

When adding a new case, implement it on **both** backends with the same contract before comparing.

### Timed metrics — db suite parity

| Case | Aligned? | Notes |
|------|----------|-------|
| `tti_ms` | Mostly | Same warm PGDATA template, `pg_isready --user 70:70`, timeout 8, stop clock before teardown. Quark uses `create`+`start`; Kata uses `ctr run -d`. Both mount PGDATA + `/dev/shm` + `/var/run/postgresql`. |
| `memory_idle_rss_mb` | Yes | Both use warm PGDATA, same tmpfs mounts, `sleep 1` after start (no pre-exec probe). RSS is per-sandbox: Quark sums `quark list` PID tree (args-match fallback); Kata filters ctr task args by sandbox ID. |
| `cpu_loop_ms` | Yes | Fixed 2M-iteration busybox shell loop; timer covers exec only (create/start and teardown outside). |
| `pgbench_tps` | Partial | Both use warm PGDATA, `sleep 3`, then one `pgbench -c1 -T5`. **Storage still differs**: Quark PGDATA on host tmpfs bind mount; Kata on devmapper snapshot — expect Kata TPS higher, not a harness artifact. |

Default Quark bench config uses plain io_uring (`UringIO=true`).

### Workloads

| Name | Image | Use |
|------|-------|-----|
| `busybox` | busybox | Baseline |
| `python` | python:3.12-slim | Interpreted TTI, network probes |
| `postgres` | postgres:16-alpine | Daemon TTI, idle RSS, pgbench TPS |
| `iperf` | networkstatic/iperf3 | Sandbox throughput |

### Compare Quark vs Kata (reference)

```python
results = lab.bench_all("light", n=5, setup=True)
lab.compare(results)
# Reference: kata/Firecracker. Quark should be faster when the harness is fair.
# tti_ms: quark 51 ms vs kata 487 ms (0.10x — comparable lifecycle)

results = lab.bench_all("db", n=3, setup=True)
lab.compare(results)
# tti_ms: quark 56 ms vs kata 15987 ms (warm PGDATA both sides)
# pgbench_tps: quark ~345 vs kata ~6250 (Kata devmapper vs Quark bind-mount tmpfs)
```

Results → `~/.keska-lab/results/*.json` (schema v2).

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `KESKA_LAB_HOST` | `lab.keska.vpn` | Lab hostname |
| `KESKA_LAB_QUARK_PROFILE` | `release` | `release` → `quark`, `debug` → `quark_d` |
| `KESKA_LAB_QUARK_EXEC` | `direct` | `direct` or `docker` |
| `KESKA_LAB_KATA_HYPERVISOR` | `firecracker` | Kata hypervisor |
| `KESKA_LAB_ENABLE_TSOT` | — | Set `1` to deploy TSOT stack during Quark network setup (required for crictl-based Quark network probes) |
| `KESKA_LAB_WORK` | `/tmp/keska-lab` | Cached OCI bundles |
| `KESKA_LAB_IO_BENCH_DIR` | `/var/lib/keska-lab/io-bench` | Host disk bind-mount for `io_read`/`io_write` (not tmpfs) |
| `KESKA_LAB_IMAGE_REGISTRY` | `europe-north1-docker.pkg.dev/keska-devops/base-images` | Pull mirror tried before Docker Hub; set empty to disable. Setup runs `image-registry-auth` (probe pull + `gcloud auth login --no-launch-browser` on the lab host if needed). Verify with `keska-lab-registry-check`. |

---

## API reference

```python
lab.quark.run()
lab.quark.bench("light", n=10, workload="busybox")
lab.kata.bench("light", n=10)
lab.bench_all("light", n=5, setup=True)
lab.compare(results)
lab.cleanup()
```
