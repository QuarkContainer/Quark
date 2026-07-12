# Keska Lab — usage

The lab is a **single-node testbed** you drive from your machine over SSH. Think of it like a small vSphere: you configure the host once, boot picoVMs (sandboxes), run benchmarks, tear them down, and only then change networking or runtime settings.

The Python package lives in [`lab/`](../../lab/). This doc covers the **node-oriented workflow** (`NodeProfile` → install → `KNode`). The older `lab.quark` / `lab.kata` paths still work for benchmarks but should not be used for host configuration.

---

## Install the client

```bash
cd lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
ssh lab@lab.keska.vpn   # key-based auth must work
```

Optional: set `KESKA_LAB_LOCAL_REPO` to your Quark checkout so rsync knows where sources live (defaults to repo root inferred from the package).

---

## Mental model

1. **Pick a profile** — frozen description of what the host should look like (`quark_bridge`, `quark_tsot`, `kata_bridge`).
2. **Install** — idempotent pipeline on the lab host; fails loudly if sandboxes are still running.
3. **Verify** — read-only health checks match profile intent (CNI type, `EnableTsot`, TSOT services, CRI).
4. **Gate** — short functional smoke (bridge: crictl lifecycle; TSOT: CreatePod → GetPodSandboxAddr).
5. **Operate** — bench, inspect status, tear down sandboxes.
6. **Cleanup profile** — stop TSOT services, revert CNI to bridge (only when no VMs are up).

You **cannot** switch network mode while picoVMs are running. `NodeInstaller` and `KNode.teardown_all()` enforce this.

---

## Profiles

| Profile | Runtime | Network | Typical use |
|---------|---------|---------|-------------|
| `quark_bridge` | Quark | bridge CNI | Default benches, CRI lifecycle |
| `quark_tsot` | Quark | TSOT stack + tsot CNI | Platform-style networking, TSOT gates |
| `kata_bridge` | Kata/Firecracker | bridge CNI | Reference comparisons |

Select via `--profile`, Python, or env:

```bash
export KESKA_LAB_NETWORK_MODE=tsot   # bridge | tsot
export KESKA_LAB_QUARK_PROFILE=release
```

`KESKA_LAB_ENABLE_TSOT=1` is **deprecated** — use `KESKA_LAB_NETWORK_MODE=tsot`.

---

## CLI (`keska-lab-node`)

```bash
# Full install for TSOT (build Quark, containerd, TSOT stack, gates)
KESKA_LAB_SKIP_REGISTRY_AUTH=1 keska-lab-node install --profile quark_tsot

# Skip Quark rebuild when binary already on host
KESKA_LAB_SKIP_REGISTRY_AUTH=1 keska-lab-node install --profile quark_tsot --skip-provision

# Reconfigure CNI / quark config / TSOT only (no containerd reprovision)
KESKA_LAB_SKIP_REGISTRY_AUTH=1 keska-lab-node install --profile quark_tsot --network-only

# Read-only checks
keska-lab-node verify --profile quark_tsot
keska-lab-node verify --profile quark_tsot --network-only   # skip CRI checks

# Functional smoke without reinstalling
keska-lab-node gate --profile quark_tsot

# Status summary
keska-lab-node status --profile quark_tsot

# Tear down TSOT services and revert CNI (no VMs running)
keska-lab-node cleanup --profile quark_tsot
```

### Install flags

| Flag | Effect |
|------|--------|
| `--skip-registry-auth` | Non-interactive; skip GCP artifact-registry login on lab |
| `--network-only` | Quark config + CNI + TSOT stack only; scoped verify; skips gates |
| `--skip-provision` | Do not rsync/build/install Quark binary |
| `--skip-containerd` | Keep existing containerd config if crictl already works |
| `--skip-gates` | Install + verify only, no L1 gate |
| `--gate L1` | Gate level (default `L1`) |

Set `KESKA_LAB_SKIP_REGISTRY_AUTH=1` in CI or when registry auth is already configured on the lab host.

---

## Python API

```python
from keska_lab import LabSession, NodeProfile

lab = LabSession(NodeProfile.quark_tsot())

# Install → verify → gate L1
node = lab.install()

# Benchmark (uses installed profile runtime; setup=False — host already prepared)
node.bench("network", n=3)

# Required before profile/cleanup changes
node.teardown_all()

# Revert TSOT / switch profile
lab.cleanup()
```

`lab.install(network_only=True)` maps to `--network-only`. Legacy benchmark entry points:

```python
lab.quark.bench("light", n=10, setup=True)
lab.bench_all("light", n=5, setup=True)
lab.compare(results)
```

Use **`node.bench`** when the host was installed via `lab.install()` so profile and runtime stay aligned.

---

## Typical workflows

### First-time TSOT on lab

```bash
KESKA_LAB_SKIP_REGISTRY_AUTH=1 keska-lab-node install --profile quark_tsot
keska-lab-node verify --profile quark_tsot
keska-lab-node gate --profile quark_tsot
```

### Switch bridge → TSOT (no VMs running)

```bash
keska-lab-node status --profile quark_bridge   # confirm vms=0
KESKA_LAB_SKIP_REGISTRY_AUTH=1 keska-lab-node install --profile quark_tsot --skip-provision
```

### Fix mode drift without touching containerd

When CNI says `tsot` but `EnableTsot=false`, or TSOT services are stale:

```bash
KESKA_LAB_SKIP_REGISTRY_AUTH=1 keska-lab-node install --profile quark_tsot --network-only
keska-lab-node verify --profile quark_tsot --network-only
```

### Back to bridge

```bash
node.teardown_all()   # Python, or crictl cleanup manually
keska-lab-node cleanup --profile quark_tsot
KESKA_LAB_SKIP_REGISTRY_AUTH=1 keska-lab-node install --profile quark_bridge --skip-provision
```

---

## Verification output

`verify` and `status` report profile-aware checks:

- **Base:** SSH, containerd socket, crictl, runtime binary
- **Alignment:** CNI conflist type, `/etc/quark/config.json` `EnableTsot`, `mode_drift` (both must match profile)
- **TSOT (`quark_tsot`):** na liveness, tsot socket, etcd container, ss on `:8890`, qlet config

A failing `mode_drift` check means the host was partially reconfigured — run `install` for the target profile rather than toggling env vars by hand.

---

## Benchmarks

Suite names and Quark-vs-Kata philosophy are documented in [`lab/README.md`](../../lab/README.md). After `lab.install()`, prefer:

```python
node.bench("light", n=10)
node.bench("network", n=3)
node.bench("db", n=3)
```

Network and TSOT suites expect the host to have been installed with the matching profile (`quark_tsot` for TSOT crictl paths that pre-register pod UIDs via `na` CreatePod).

---

## Environment reference

| Variable | Default | Notes |
|----------|---------|-------|
| `KESKA_LAB_HOST` | `lab.keska.vpn` | SSH target |
| `KESKA_LAB_NETWORK_MODE` | `bridge` | `bridge`, `tsot` |
| `KESKA_LAB_QUARK_PROFILE` | `release` | `release` → `quark`, `debug` → `quark_d` |
| `KESKA_LAB_SKIP_REGISTRY_AUTH` | — | Set `1` to skip registry login step |
| `KESKA_LAB_LOCAL_REPO` | auto | Path to Quark checkout on your machine |
| `KESKA_LAB_SSH_KEY` | agent | Optional explicit key path |

Full list: [`lab/README.md#environment-variables`](../../lab/README.md#environment-variables).

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `NodeBusy: N sandbox(es) still running` | picoVMs still up | `node.teardown_all()` or manual crictl cleanup |
| `mode_drift` on verify | CNI vs `EnableTsot` mismatch | `install --profile <target>` |
| `crictl info` fails | containerd CRI plugin not loaded | Full install; Quark profiles must not pull in kata/devmapper (see architecture doc) |
| TSOT gate CreatePod JSON error | Stale gate pod def | Update lab package; gate uses structured `PodStatus` |
| `ss not ready on 8890` | etcd not up before ss | Re-run install; ss waits for etcd health |
| na exits after containerd restart | Expected if stack started before CRI | Install order: containerd first, TSOT stack after |

Bug notes: [`kdoc/bugs/`](../bugs/). Lab-specific ops snippets: [`kdoc/ops/`](../ops/).
