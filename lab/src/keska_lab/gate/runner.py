"""Run profile-aware install gates on the lab host."""

from __future__ import annotations

import time

from keska_lab.gate.base_gate import GateResult
from keska_lab.gate.bridge_gate import bridge_gate_l1_script
from keska_lab.gate.tsot_gate import tsot_gate_l1_script
from keska_lab.profile import NetworkMode, NodeProfile
from keska_lab.remote import RemoteHost
from keska_lab.setup.containerd_cri import cri_smoke_script


class NodeGateRunner:
    GATE_TIMEOUT = {
        "base_l1": 60,
        "bridge_l1": 300,
        "tsot_l1": 300,
    }

    def __init__(self, remote: RemoteHost, profile: NodeProfile):
        self.remote = remote
        self.profile = profile

    def run(self, level: str = "L1") -> list[GateResult]:
        level = level.upper()
        results: list[GateResult] = []
        results.append(self._run_script("base_l1", cri_smoke_script()))
        if level in ("L1", "L2"):
            if self.profile.network == NetworkMode.tsot:
                repo = self.remote.config.remote_repo
                script = tsot_gate_l1_script(
                    repo,
                    pod_mgr_port=self.profile.net_params.pod_mgr_port,
                    tsot_cni_port=self.profile.net_params.tsot_cni_port,
                )
                results.append(self._run_script("tsot_l1", script))
            else:
                handler = self.profile.runtime if self.profile.runtime == "kata" else "quark"
                results.append(
                    self._run_script("bridge_l1", bridge_gate_l1_script(runtime_handler=handler))
                )
        return results

    def _run_script(self, name: str, script: str) -> GateResult:
        t0 = time.perf_counter()
        timeout = self.GATE_TIMEOUT.get(name, 120)
        r = self.remote.sh(script, timeout=timeout)
        duration = time.perf_counter() - t0
        if not r.ok:
            msg = self.remote.format_failure(r)
            return GateResult(name, False, msg, duration)
        lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        msg = lines[-1] if lines else f"{name} ok"
        return GateResult(name, True, msg, duration)
