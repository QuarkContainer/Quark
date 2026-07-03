"""CNI plugin setup for containerd CRI (lab host only)."""

from __future__ import annotations

import textwrap

from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupStep, StepResult

CNI_VERSION = "v1.4.0"
CNI_URL = (
    f"https://github.com/containernetworking/plugins/releases/download/"
    f"{CNI_VERSION}/cni-plugins-linux-amd64-{CNI_VERSION}.tgz"
)
CONFLIST = "/etc/cni/net.d/10-containerd-net.conflist"


def cni_forward_script() -> str:
    """Allow pod egress when host FORWARD policy is DROP (common with Docker)."""
    return textwrap.dedent(
        """
        set -euo pipefail
        if ! sudo -n iptables -C FORWARD -o cni0 -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT 2>/dev/null; then
          sudo -n iptables -I FORWARD 1 -o cni0 -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
          sudo -n iptables -I FORWARD 2 -i cni0 -j ACCEPT
          echo "cni0 forward rules installed"
        else
          echo "cni0 forward rules present"
        fi
        """
    ).strip()


def cni_install_script() -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        if [ -x /opt/cni/bin/bridge ] && [ -f {CONFLIST} ]; then
          echo "cni plugins already installed"
          exit 0
        fi
        tmp=$(mktemp -d)
        trap 'rm -rf "$tmp"' EXIT
        curl -fsSL {CNI_URL!r} -o "$tmp/cni.tgz"
        sudo -n mkdir -p /opt/cni/bin /etc/cni/net.d
        sudo -n tar -C /opt/cni/bin -xzf "$tmp/cni.tgz"
        sudo -n tee {CONFLIST} >/dev/null <<'JSON'
{{
  "cniVersion": "1.0.0",
  "name": "containerd-net",
  "plugins": [
    {{
      "type": "bridge",
      "bridge": "cni0",
      "isGateway": true,
      "ipMasq": true,
      "promiscMode": true,
      "ipam": {{
        "type": "host-local",
        "ranges": [[{{ "subnet": "10.22.0.0/16" }}]],
        "routes": [{{ "dst": "0.0.0.0/0" }}]
      }}
    }},
    {{
      "type": "portmap",
      "capabilities": {{ "portMappings": true }}
    }}
  ]
}}
JSON
        test -x /opt/cni/bin/bridge
        """
    ).strip()


class CniPluginsStep(SetupStep):
    name = "cni-plugins"

    def run(self, remote: RemoteHost) -> StepResult:
        r = remote.sh(cni_install_script(), timeout=300)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        fwd = remote.sh(cni_forward_script(), timeout=60)
        if not fwd.ok:
            return StepResult(self.name, False, remote.format_failure(fwd))
        return StepResult(self.name, True, "cni bridge + conflist ready")
