# 013 — CRI stats lab blocked by crictl v1.31 + containerd config

## Symptom

Lab H2 (`crictl stats`) failed with `load podSandboxConfig: config at <pod-id> not found` before Quark stats code ran. Same error with `runtime_handler: runc`.

## Cause

1. **crictl v1.31** changed `crictl run` usage: second argument is **pod-config.json**, not pod sandbox ID (from `runp`). Passing a pod ID made crictl try to open that hex string as a file.
2. **containerd 2.x** on lab needed a clean `version = 3` config from `containerd config default` plus `overlayfs` `unpack_config` for image pull; appended legacy `version = 2` / `grpc.v1.cri` fragments broke the config file.

## Fix

- `lab/src/keska_lab/setup/containerd_cri.py`: regenerate v3 CRI config, pre-pull images via `ctr`, use `crictl run --no-pull container.json pod.json`, parse stats table output.
- `lab/src/keska_lab/setup/quark_config.py`: `cri_bench_config_json()` (`ShimMode`, `Sandboxed`, cgroups on).
- `QuarkCriStatsStep` wired into heavy/network setup pipelines.

## Files

- `lab/src/keska_lab/setup/containerd_cri.py`
- `lab/src/keska_lab/setup/quark_config.py`
- `lab/src/keska_lab/setup/pipelines.py`

## Verify

```bash
cd lab && python3 -c "
from keska_lab.remote import RemoteHost
from keska_lab.setup.containerd_cri import QuarkCriStatsStep
r = QuarkCriStatsStep().run(RemoteHost(), stream=True)
assert r.ok, r.message
print(r.message)
"
```

Expected: `H2 PASS: crictl stats` with non-zero `memory_usage`.
