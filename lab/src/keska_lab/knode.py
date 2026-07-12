"""Operational interface for a configured lab node."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

from keska_lab.config import LabConfig
from keska_lab.health import HealthReport, NodeHealthChecker
from keska_lab.installer import InstallOptions, NodeInstaller, list_running_vms
from keska_lab.profile import NodeProfile
from keska_lab.remote import RemoteHost
from keska_lab.runtime import KataEnvironment, QuarkEnvironment


@dataclass
class NodeStatus:
    profile: NodeProfile
    vm_count: int
    health: HealthReport

    @property
    def ok(self) -> bool:
        return self.health.ok and self.vm_count >= 0


class KNode:
    def __init__(self, remote: RemoteHost, profile: NodeProfile):
        self.remote = remote
        self.profile = profile
        self.config = remote.config

    @classmethod
    def install(
        cls,
        remote: RemoteHost,
        profile: NodeProfile,
        *,
        gate_level: str = "L1",
        options: InstallOptions | None = None,
    ) -> KNode:
        opts = options or InstallOptions(gate_level=gate_level)
        if options is None:
            opts.gate_level = gate_level
        return NodeInstaller(remote, profile).install(options=opts)

    def status(self) -> NodeStatus:
        return NodeStatus(
            profile=self.profile,
            vm_count=len(self.running_vms()),
            health=self.verify(),
        )

    def verify(self) -> HealthReport:
        return NodeHealthChecker(self.remote, self.profile).verify()

    def running_vms(self) -> list[str]:
        return list_running_vms(self.remote)

    def teardown_all(self) -> None:
        script = textwrap.dedent(
            """
            set -euo pipefail
            for id in $(sudo -n crictl ps -q 2>/dev/null); do
              sudo -n crictl rm -f "$id" 2>/dev/null || true
            done
            for id in $(sudo -n crictl pods -q 2>/dev/null); do
              sudo -n crictl stopp "$id" 2>/dev/null || true
              sudo -n crictl rmp "$id" 2>/dev/null || true
            done
            """
        ).strip()
        self.remote.sh(script, timeout=300)
        remaining = self.running_vms()
        if remaining:
            raise NodeBusy(remaining)

    def bench(self, suite: str, **kwargs):
        self._assert_ready()
        env = self._runtime_env()
        return env.bench(suite, setup=False, **kwargs)

    def _assert_ready(self) -> None:
        health = self.verify()
        if health.failed:
            names = ", ".join(c.name for c in health.failed)
            raise RuntimeError(f"node not ready: {names}")

    def _runtime_env(self):
        if self.profile.runtime == "kata":
            return KataEnvironment(self.remote, self.config)
        return QuarkEnvironment(self.remote, self.config)


def install_node(
    config: LabConfig | None = None,
    profile: NodeProfile | None = None,
    *,
    gate_level: str = "L1",
    options: InstallOptions | None = None,
) -> KNode:
    cfg = config or LabConfig.from_env()
    prof = profile or cfg.node_profile
    remote = RemoteHost(cfg)
    opts = options or InstallOptions(gate_level=gate_level)
    if options is None:
        opts.gate_level = gate_level
    return KNode.install(remote, prof, options=opts)
