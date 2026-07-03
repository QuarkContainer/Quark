"""Kata + devmapper prerequisites for CRI gate (runp/create/start path)."""

from __future__ import annotations

import textwrap

from keska_lab.config import LabConfig
from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupStep, StepResult
from keska_lab.setup.cni import CniPluginsStep
from keska_lab.setup.docker import KataInstallStep
from keska_lab.setup.kata_firecracker import KataHypervisorStep


def cri_kata_pull_images_script() -> str:
    """Pre-pull pause + busybox into kata/devmapper snapshotter via CRI."""
    return textwrap.dedent(
        """
        set -euo pipefail
        sudo crictl pull --runtime=kata registry.k8s.io/pause:3.8
        sudo crictl pull --runtime=kata docker.io/library/busybox:latest
        echo "kata cri images ready"
        """
    ).strip()


def kata_cri_health_script() -> str:
    """Return 0 when devmapper pool + kata ctr smoke work."""
    return textwrap.dedent(
        """
        set -euo pipefail
        grep -q "pool_name = 'devpool'" /etc/containerd/config.toml
        sudo -n ctr plugins ls | grep -F devmapper | grep -q ' ok '
        sudo -n dmsetup info devpool >/dev/null
        test -x /opt/kata/bin/kata-runtime
        timeout 45 sudo -n ctr run --rm --runtime io.containerd.kata.v2 --snapshotter devmapper \\
          docker.io/library/busybox:latest keska-kata-cri-health-$RANDOM /bin/true
        """
    ).strip()


class CriKataImagesStep(SetupStep):
    name = "cri-kata-images"

    def run(self, remote: RemoteHost) -> StepResult:
        r = remote.sh(cri_kata_pull_images_script(), timeout=300)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        msg = r.stdout.strip().splitlines()[-1]
        return StepResult(self.name, True, msg)


def ensure_kata_cri_ready(
    remote: RemoteHost,
    config: LabConfig | None = None,
) -> list[StepResult]:
    """Install Kata + devmapper + CNI + CRI images if not already healthy."""
    cfg = config or remote.config
    health = remote.sh(kata_cri_health_script(), timeout=120)
    if health.ok:
        return [StepResult("kata-cri-ready", True, "already healthy")]

    results: list[StepResult] = []
    for step in (
        KataInstallStep(),
        KataHypervisorStep(cfg),
        CniPluginsStep(),
        CriKataImagesStep(),
    ):
        res = step.run(remote)
        results.append(res)
        if not res.ok:
            break
    return results
