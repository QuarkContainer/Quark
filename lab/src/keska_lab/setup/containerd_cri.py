"""Enable containerd CRI + Quark shim runtimes (lab host only)."""

from __future__ import annotations

import textwrap

from keska_lab.harness.network import crictl_config_script
from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupStep, StepResult

MARKER = "keska-lab cri"
OLD_MARKER = "keska-lab quark runtimes"
CRICTL_VERSION = "v1.31.1"
CRICTL_URL = (
    f"https://github.com/kubernetes-sigs/cri-tools/releases/download/"
    f"{CRICTL_VERSION}/crictl-{CRICTL_VERSION}-linux-amd64.tar.gz"
)


def containerd_cri_config_script() -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        CFG=/etc/containerd/config.toml
        sudo -n python3 <<'PY'
import re
from pathlib import Path

cfg = Path("/etc/containerd/config.toml")
text = cfg.read_text()
text = re.sub(r'^disabled_plugins.*\\n', '', text, flags=re.M)
for marker in ({OLD_MARKER!r}, {MARKER!r}):
    text = re.sub(rf"\\n# {{re.escape(marker)}}.*?(?=\\n# |\\Z)", "", text, flags=re.S)
cfg.write_text(text.rstrip() + "\\n")
print("stripped old cri blocks")
PY
        sudo -n tee -a "$CFG" >/dev/null <<'TOML'

# {MARKER}
version = 2
[plugins."io.containerd.grpc.v1.cri"]
  sandbox_image = "registry.k8s.io/pause:3.8"
  [plugins."io.containerd.grpc.v1.cri".cni]
    bin_dir = "/opt/cni/bin"
    conf_dir = "/etc/cni/net.d"
[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "quark"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
    runtime_type = "io.containerd.runc.v2"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.quark]
    runtime_type = "io.containerd.quark.v1"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.quarkd]
    runtime_type = "io.containerd.quarkd.v1"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.kata]
    runtime_type = "io.containerd.kata.v2"
TOML
        sudo -n systemctl restart containerd
        sleep 4
        sudo -n ctr plugins ls | grep -F 'io.containerd.grpc.v1' | grep -w 'cri' | grep -q ' ok '
        sudo -n chmod 666 /run/containerd/containerd.sock
        """
    ).strip()


def cri_smoke_script() -> str:
    return textwrap.dedent(
        """
        set -euo pipefail
        if command -v crictl >/dev/null 2>&1; then
          sudo -n crictl info >/dev/null
          echo "crictl ok"
          exit 0
        fi
        sudo -n ctr --namespace k8s.io containers ls >/dev/null 2>&1 || true
        test -S /run/containerd/containerd.sock
        echo "containerd cri socket ok"
        """
    ).strip()


class CrictlInstallStep(SetupStep):
    name = "crictl-install"

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            if ! command -v crictl >/dev/null 2>&1; then
              tmp=$(mktemp -d)
              curl -fsSL {CRICTL_URL!r} -o "$tmp/crictl.tgz"
              sudo -n tar -C /usr/local/bin -xzf "$tmp/crictl.tgz" crictl
              rm -rf "$tmp"
            fi
            {crictl_config_script()}
            crictl --version | head -1
            """
        ).strip()
        r = remote.sh(script, timeout=120, stream=stream)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        ver = r.stdout.strip().splitlines()[-1]
        return StepResult(self.name, True, ver)


class ContainerdCriStep(SetupStep):
    name = "containerd-cri"

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        r = remote.sh(containerd_cri_config_script(), timeout=180, stream=stream)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        smoke = remote.sh(cri_smoke_script(), timeout=60, stream=stream)
        if not smoke.ok:
            return StepResult(self.name, False, remote.format_failure(smoke))
        msg = smoke.stdout.strip().splitlines()[-1] if smoke.stdout.strip() else "cri enabled"
        return StepResult(self.name, True, msg)
