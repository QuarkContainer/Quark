"""Read-only health probes for an installed lab node."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from enum import Enum

from keska_lab.profile import NetworkMode, NodeProfile
from keska_lab.remote import RemoteHost
from keska_lab.setup.cni import active_cni_type_script, cni_conflist_type_script
from keska_lab.setup.quark_config import QUARK_CONFIG, quark_config_enable_tsot_script
from keska_lab.setup.tsot import QLET_CONFIG_PATH, TSOT_SOCKET, na_liveness_script


class ServiceStatus(str, Enum):
    ok = "ok"
    degraded = "degraded"
    fail = "fail"
    skip = "skip"


@dataclass
class CheckResult:
    name: str
    status: ServiceStatus
    message: str


@dataclass
class HealthReport:
    profile: NodeProfile
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status in (ServiceStatus.ok, ServiceStatus.skip, ServiceStatus.degraded) for c in self.checks)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == ServiceStatus.fail]

    def summary(self) -> str:
        parts = [f"{c.name}={c.status.value}" for c in self.checks if c.status == ServiceStatus.fail]
        return "; ".join(parts) if parts else "ok"


class NodeHealthChecker:
    def __init__(self, remote: RemoteHost, profile: NodeProfile):
        self.remote = remote
        self.profile = profile

    def verify(self, *, network_only: bool = False) -> HealthReport:
        report = HealthReport(profile=self.profile)
        report.checks.append(self._check_ssh())
        if network_only:
            report.checks.append(self._check_cni_conflist())
            if self.profile.runtime == "quark":
                report.checks.append(self._check_quark_config())
            report.checks.append(self._check_mode_drift())
            if self.profile.network == NetworkMode.tsot:
                report.checks.extend(self._check_tsot_stack())
            else:
                report.checks.append(self._check_na_stopped())
            return report
        report.checks.append(self._check_containerd())
        report.checks.append(self._check_crictl())
        report.checks.append(self._check_runtime_binary())
        report.checks.append(self._check_cni_conflist())
        report.checks.append(self._check_quark_config())
        report.checks.append(self._check_mode_drift())
        if self.profile.network == NetworkMode.tsot:
            report.checks.extend(self._check_tsot_stack())
        else:
            report.checks.append(self._check_na_stopped())
        return report

    def _run(self, script: str, *, timeout: int = 30) -> tuple[bool, str]:
        r = self.remote.sh(script, timeout=timeout)
        msg = (r.stdout or r.stderr or "").strip().splitlines()
        last = msg[-1] if msg else f"exit {r.returncode}"
        return r.ok, last

    def _check_ssh(self) -> CheckResult:
        ok, msg = self._run("echo SSH_OK")
        return CheckResult("ssh_reachable", ServiceStatus.ok if ok else ServiceStatus.fail, msg)

    def _check_containerd(self) -> CheckResult:
        ok, msg = self._run("test -S /run/containerd/containerd.sock && echo ok")
        return CheckResult(
            "containerd_socket",
            ServiceStatus.ok if ok else ServiceStatus.fail,
            msg,
        )

    def _check_crictl(self) -> CheckResult:
        ok, msg = self._run("sudo -n crictl info >/dev/null 2>&1 && echo ok")
        return CheckResult("crictl_info", ServiceStatus.ok if ok else ServiceStatus.fail, msg)

    def _check_runtime_binary(self) -> CheckResult:
        if self.profile.runtime == "kata":
            ok, msg = self._run("command -v containerd-shim-kata-v2 >/dev/null && echo ok")
            return CheckResult("kata_runtime", ServiceStatus.ok if ok else ServiceStatus.fail, msg)
        bin_name = "quark_d" if self.profile.build_profile == "debug" else "quark"
        ok, msg = self._run(f"command -v {bin_name} >/dev/null && echo ok")
        return CheckResult("quark_binary", ServiceStatus.ok if ok else ServiceStatus.fail, msg)

    def _check_cni_conflist(self) -> CheckResult:
        expected = "tsot" if self.profile.network == NetworkMode.tsot else "bridge"
        ok, msg = self._run(cni_conflist_type_script(expected))
        return CheckResult(
            "cni_conflist_type",
            ServiceStatus.ok if ok else ServiceStatus.fail,
            f"expected {expected}: {msg}",
        )

    def _check_quark_config(self) -> CheckResult:
        if self.profile.runtime != "quark":
            return CheckResult("quark_config_enable_tsot", ServiceStatus.skip, "n/a")
        expected = self.profile.network == NetworkMode.tsot
        ok, msg = self._run(quark_config_enable_tsot_script(expected))
        return CheckResult(
            "quark_config_enable_tsot",
            ServiceStatus.ok if ok else ServiceStatus.fail,
            f"EnableTsot={expected}: {msg}",
        )

    def _check_mode_drift(self) -> CheckResult:
        expected_net = self.profile.network.value
        expected_tsot = "true" if self.profile.network == NetworkMode.tsot else "false"
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            EXP_NET={json.dumps(expected_net)}
            EXP_TSOT={expected_tsot}
            ACT_NET=$({active_cni_type_script()})
            ACT_TSOT=$(python3 -c "import json; print(str(json.load(open('{QUARK_CONFIG}')).get('EnableTsot', False)).lower())" 2>/dev/null || echo unknown)
            if [ "$ACT_NET" != "$EXP_NET" ] || [ "$ACT_TSOT" != "$EXP_TSOT" ]; then
              echo "drift net=$ACT_NET tsot=$ACT_TSOT expected net=$EXP_NET tsot=$EXP_TSOT"
              exit 1
            fi
            echo ok
            """
        ).strip()
        ok, msg = self._run(script, timeout=45)
        return CheckResult("mode_drift", ServiceStatus.ok if ok else ServiceStatus.fail, msg)

    def _check_tsot_stack(self) -> list[CheckResult]:
        checks: list[CheckResult] = []
        ok, msg = self._run(na_liveness_script())
        checks.append(CheckResult("na_liveness", ServiceStatus.ok if ok else ServiceStatus.fail, msg))
        ok, msg = self._run(f"test -S {TSOT_SOCKET} && echo ok")
        checks.append(CheckResult("tsot_socket", ServiceStatus.ok if ok else ServiceStatus.fail, msg))
        ok, msg = self._run(
            "sudo -n docker ps --format '{{.Names}}' 2>/dev/null | grep -qx keska-etcd && echo ok || echo missing"
        )
        etcd_ok = ok and msg.strip() == "ok"
        checks.append(
            CheckResult(
                "etcd_running",
                ServiceStatus.ok if etcd_ok else ServiceStatus.fail,
                msg,
            )
        )
        ok, msg = self._run(
            f"/bin/ss -lntp 2>/dev/null | grep -q ':{self.profile.net_params.state_svc_port}' && echo ok || echo missing"
        )
        ss_ok = ok and msg.strip() == "ok"
        checks.append(
            CheckResult(
                "ss_port",
                ServiceStatus.ok if ss_ok else ServiceStatus.degraded,
                msg,
            )
        )
        ok, msg = self._run(f"test -f {QLET_CONFIG_PATH} && echo ok")
        checks.append(CheckResult("qlet_config", ServiceStatus.ok if ok else ServiceStatus.fail, msg))
        return checks

    def _check_na_stopped(self) -> CheckResult:
        ok, msg = self._run("pgrep -x na >/dev/null && echo running || echo stopped")
        if ok and msg == "stopped":
            return CheckResult("na_stopped", ServiceStatus.ok, msg)
        if ok and msg == "running":
            return CheckResult("na_stopped", ServiceStatus.degraded, "na running on bridge profile")
        return CheckResult("na_stopped", ServiceStatus.fail, msg)
