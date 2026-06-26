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
| `light` / `full` | TTI, TTI under load, memory, pause/resume | Quark + Kata |
| `workloads` | TTI + memory idle per workload | Quark + Kata |
| `standard` | `light` on busybox + `workloads` on python | Quark + Kata |
| `network` | inet connect/download, sandbox iperf (CRI) | Quark + Kata |
| `db` | postgres TTI, idle RSS, pgbench TPS | Quark + Kata |
| `heavy` | light + io_fs + network (legacy) | Mixed |

### Workloads

| Name | Image | Use |
|------|-------|-----|
| `busybox` | busybox | Baseline |
| `python` | python:3.12-slim | Interpreted TTI, network probes |
| `postgres` | postgres:16-alpine | Daemon TTI, idle RSS, pgbench TPS |
| `iperf` | networkstatic/iperf3 | Sandbox throughput |

### Compare Quark vs Kata

```python
results = lab.bench_all("light", n=5, setup=True)
lab.compare(results)
# tti_ms: quark 61.0 vs kata 497.0 ms  (0.12x)

results = lab.bench_all("db", n=3, setup=True)
lab.compare(results)
# tti_ms: quark ~50ms vs kata ~TBD; memory_idle_rss_mb; pgbench_tps ~300+ on Quark
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
| `KESKA_LAB_IMAGE_REGISTRY` | `europe-north1-docker.pkg.dev/keska-devops/base-images` | Pull mirror tried before Docker Hub; set empty to disable. Setup runs `image-registry-auth` (probe pull + `gcloud auth login` if needed). Verify with `keska-lab-registry-check`. |

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
