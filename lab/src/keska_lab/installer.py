"""Profile-driven host installation and cleanup."""

from __future__ import annotations

import textwrap

from dataclasses import dataclass

from keska_lab.config import LabConfig
from keska_lab.gate.runner import NodeGateRunner
from keska_lab.health import NodeHealthChecker
from keska_lab.profile import NetworkMode, NodeProfile
from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupPipeline, SetupStep, StepResult
from keska_lab.setup.cni import CniPluginsStep
from keska_lab.setup.containerd_cri import ContainerdCriStep, CrictlInstallStep, QuarkCriStatsStep
from keska_lab.setup.docker import DockerEnsureStep, KataInstallStep
from keska_lab.setup.image_registry import ImageRegistryAuthStep
from keska_lab.setup.kata_firecracker import KataHypervisorStep
from keska_lab.setup.pipelines import CleanupSandboxesStep
from keska_lab.setup.quark_cleanup import cleanup_quark_sandboxes
from keska_lab.setup.quark_config import QuarkConfigStep
from keska_lab.setup.tsot import TsotStackStep, TsotStackStopStep


class InstallError(RuntimeError):
    def __init__(self, step: str, message: str):
        super().__init__(f"{step}: {message}")
        self.step = step
        self.message = message


class NodeBusy(RuntimeError):
    def __init__(self, running_vms: list[str]):
        self.running_vms = running_vms
        super().__init__(f"{len(running_vms)} sandbox(es) still running: {running_vms[:5]}")


def running_vms_script() -> str:
    return textwrap.dedent(
        """
        set -euo pipefail
        sudo -n crictl pods --state Ready 2>/dev/null | awk 'NR>1 {print $1}' || true
        """
    ).strip()


def list_running_vms(remote: RemoteHost) -> list[str]:
    r = remote.sh(running_vms_script(), timeout=30)
    if not r.ok:
        return []
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


class ProvisionQuarkStep(SetupStep):
    name = "provision-quark"

    def __init__(self, config: LabConfig, profile: NodeProfile):
        self.config = config
        self.profile = profile

    def run(self, remote: RemoteHost) -> StepResult:
        from keska_lab.provision import provision_quark

        self.config.quark_build_profile = self.profile.build_profile
        report = provision_quark(remote, self.config)
        if not report.ok:
            failed = next(s for s in report.steps if not s.ok)
            return StepResult(self.name, False, f"{failed.name}: {failed.message}")
        return StepResult(self.name, True, "quark provisioned")


class CrictlReadyStep(SetupStep):
    """Skip containerd reconfiguration when CRI already answers."""

    name = "crictl-ready"

    def run(self, remote: RemoteHost) -> StepResult:
        r = remote.sh("sudo -n crictl info >/dev/null 2>&1 && echo ok", timeout=30)
        if r.ok and "ok" in r.stdout:
            return StepResult(self.name, True, "crictl already ready")
        return StepResult(self.name, False, "crictl not ready — run full install")


@dataclass
class InstallOptions:
    gate_level: str = "L1"
    network_only: bool = False
    skip_provision: bool = False
    skip_containerd: bool = False
    skip_gates: bool = False


class NodeInstaller:
    def __init__(self, remote: RemoteHost, profile: NodeProfile):
        self.remote = remote
        self.profile = profile
        self.config = remote.config

    def install(self, *, options: InstallOptions | None = None):
        from keska_lab.knode import KNode

        opts = options or InstallOptions()
        self.profile.validate()
        running = list_running_vms(self.remote)
        if running:
            raise NodeBusy(running)

        pipeline = self._install_pipeline(opts)
        report = pipeline.run(self.remote)
        if not report.ok:
            failed = next(s for s in report.steps if not s.ok)
            raise InstallError(failed.name, failed.message)

        health = self.verify(network_only=opts.network_only)
        if health.failed:
            raise InstallError("verify", health.summary())

        if not opts.skip_gates and not opts.network_only:
            gates = NodeGateRunner(self.remote, self.profile).run(opts.gate_level)
            failed_gates = [g for g in gates if not g.ok]
            if failed_gates:
                raise InstallError(failed_gates[0].gate, failed_gates[0].message)

        return KNode(self.remote, self.profile)

    def cleanup(self) -> None:
        running = list_running_vms(self.remote)
        if running:
            raise NodeBusy(running)
        pipeline = self._cleanup_pipeline()
        report = pipeline.run(self.remote)
        if not report.ok:
            failed = next(s for s in report.steps if not s.ok)
            raise InstallError(failed.name, failed.message)
        cleanup_quark_sandboxes(self.remote)

    def verify(self, *, network_only: bool = False):
        return NodeHealthChecker(self.remote, self.profile).verify(network_only=network_only)

    def _install_pipeline(self, opts: InstallOptions) -> SetupPipeline:
        name = f"install-{self.profile.name}"
        if opts.network_only:
            name += "-network"
        pipe = SetupPipeline(name)
        pipe.add(CleanupSandboxesStep())

        if opts.network_only:
            if self.profile.runtime == "quark":
                pipe.add(QuarkConfigStep(self.profile))
            pipe.add(TsotStackStopStep())
            pipe.add(CniPluginsStep(self.profile.network))
            pipe.add(CrictlInstallStep())
            if self.profile.network == NetworkMode.tsot:
                pipe.add(TsotStackStep(self.profile))
            return pipe

        pipe.add(DockerEnsureStep())
        if not self.config.skip_registry_auth:
            pipe.add(ImageRegistryAuthStep(self.config))

        if self.profile.runtime == "quark":
            if not opts.skip_provision:
                pipe.add(ProvisionQuarkStep(self.config, self.profile))
            pipe.add(QuarkConfigStep(self.profile))
        else:
            pipe.add(KataInstallStep())
            pipe.add(KataHypervisorStep(self.config))

        pipe.add(TsotStackStopStep())
        pipe.add(CniPluginsStep(self.profile.network))

        if opts.skip_containerd:
            pipe.add(CrictlReadyStep())
            pipe.add(CrictlInstallStep())
        else:
            skip_lifecycle = self.profile.network == NetworkMode.tsot
            pipe.add(ContainerdCriStep(include_kata=self.profile.runtime == "kata", skip_lifecycle_smoke=skip_lifecycle))
            pipe.add(CrictlInstallStep())

        if self.profile.network == NetworkMode.tsot:
            pipe.add(TsotStackStep(self.profile))

        if not opts.skip_containerd and self.profile.runtime == "quark":
            if self.profile.network != NetworkMode.tsot:
                pipe.add(QuarkCriStatsStep())

        return pipe

    def _cleanup_pipeline(self) -> SetupPipeline:
        pipe = SetupPipeline(f"cleanup-{self.profile.name}")
        pipe.add(TsotStackStopStep())
        if self.profile.network == NetworkMode.tsot:
            pipe.add(CniPluginsStep(NetworkMode.bridge))
        return pipe
