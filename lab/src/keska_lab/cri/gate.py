"""Run CRI validation layers on the lab host."""

from __future__ import annotations

from dataclasses import dataclass

from keska_lab.cri.api_spot import cri_api_spot_script
from keska_lab.cri.lifecycle import cri_lifecycle_smoke_script
from keska_lab.cri.multi_pod import cri_multi_container_smoke_script
from keska_lab.remote import RemoteHost
from keska_lab.setup.containerd_cri import cri_smoke_script
from keska_lab.setup.quark_config import cri_bench_config_json, deploy_config_script


@dataclass
class GateResult:
    layer: str
    ok: bool
    message: str


LAYER_SCRIPTS = {
    "L1": lambda runtime: cri_smoke_script(),
    "L2": lambda runtime: cri_lifecycle_smoke_script(runtime_handler=runtime),
    "L3": lambda runtime: cri_api_spot_script(runtime_handler=runtime),
    "L5": lambda runtime: cri_multi_container_smoke_script(runtime_handler=runtime),
}

LAYER_TIMEOUT = {
    "L1": 60,
    "L2": 300,
    "L3": 240,
    "L5": 360,
}


def runtime_flag(runtime: str) -> str:
    """Map CLI runtime name to crictl --runtime handler."""
    if runtime in ("", "default", "quark"):
        return ""
    return runtime


def run_layer(
    remote: RemoteHost,
    layer: str,
    *,
    runtime: str = "quark",
    deploy_quark_cri_config: bool = True,
    stream: bool = False,
) -> GateResult:
    layer = layer.upper()
    if layer not in LAYER_SCRIPTS:
        return GateResult(layer, False, f"unknown layer {layer}; use L1,L2,L3,L5")

    if deploy_quark_cri_config and runtime in ("quark", "default", ""):
        cfg = cri_bench_config_json()
        r = remote.sh(deploy_config_script(cfg), timeout=60, stream=stream)
        if not r.ok:
            return GateResult(layer, False, remote.format_failure(r))

    rt = runtime_flag(runtime)
    script = LAYER_SCRIPTS[layer](rt)
    timeout = LAYER_TIMEOUT[layer]
    r = remote.sh(script, timeout=timeout, stream=stream)
    if not r.ok:
        return GateResult(layer, False, remote.format_failure(r))
    msg = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"{layer} ok"
    return GateResult(layer, True, msg)


def run_layers_through(
    remote: RemoteHost,
    max_layer: str,
    *,
    runtime: str = "quark",
    deploy_quark_cri_config: bool = True,
    stream: bool = False,
) -> list[GateResult]:
    order = ["L1", "L2", "L3", "L5"]
    max_layer = max_layer.upper()
    if max_layer not in order:
        raise ValueError(f"max_layer must be one of {order}")
    stop_idx = order.index(max_layer)
    results: list[GateResult] = []
    for layer in order[: stop_idx + 1]:
        res = run_layer(
            remote,
            layer,
            runtime=runtime,
            deploy_quark_cri_config=deploy_quark_cri_config,
            stream=stream,
        )
        results.append(res)
        if not res.ok:
            break
    return results
