"""CNI plugin setup for containerd CRI (lab host only)."""

from __future__ import annotations

import json
import textwrap

from keska_lab.profile import NetworkMode
from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupStep, StepResult

CNI_VERSION = "v1.4.0"
CNI_URL = (
    f"https://github.com/containernetworking/plugins/releases/download/"
    f"{CNI_VERSION}/cni-plugins-linux-amd64-{CNI_VERSION}.tgz"
)
BRIDGE_CONFLIST = "/etc/cni/net.d/10-containerd-net.conflist"
TSOT_CONFLIST = "/etc/cni/net.d/10-tsot-net.conflist"


def bridge_conflist_body() -> str:
    return textwrap.dedent(
        """
        {
          "cniVersion": "1.0.0",
          "name": "containerd-net",
          "plugins": [
            {
              "type": "bridge",
              "bridge": "cni0",
              "isGateway": true,
              "ipMasq": true,
              "promiscMode": true,
              "ipam": {
                "type": "host-local",
                "ranges": [[{ "subnet": "10.22.0.0/16" }]],
                "routes": [{ "dst": "0.0.0.0/0" }]
              }
            },
            {
              "type": "portmap",
              "capabilities": { "portMappings": true }
            }
          ]
        }
        """
    ).strip()


def tsot_conflist_body() -> str:
    return textwrap.dedent(
        """
        {
          "cniVersion": "1.0.0",
          "name": "tsot-net",
          "plugins": [
            {
              "type": "tsot"
            }
          ]
        }
        """
    ).strip()


def active_cni_type_script() -> str:
    """Print active CNI type to stdout."""
    return textwrap.dedent(
        """
        python3 - <<'PY'
import json, glob
for p in sorted(glob.glob("/etc/cni/net.d/*.conflist")):
    c = json.load(open(p))
    for plug in c.get("plugins", []):
        t = plug.get("type", "")
        if t in ("bridge", "tsot"):
            print(t)
            raise SystemExit
print("none")
PY
        """
    ).strip()


def cni_conflist_type_script(expected: str = "") -> str:
    """Return active CNI plugin type (bridge|tsot|none). If expected set, exit 0 on match."""
    exp_check = ""
    if expected:
        exp_check = f"""
        if [ "$TYPE" != "{expected}" ]; then
          echo "expected {expected} got $TYPE"
          exit 1
        fi
        """
    return textwrap.dedent(
        f"""
        set -euo pipefail
        TYPE=$({active_cni_type_script()})
        echo "$TYPE"
        {exp_check}
        echo ok
        """
    ).strip()


def cni_install_binaries_script() -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        if [ -x /opt/cni/bin/bridge ]; then
          echo "cni plugins already installed"
          exit 0
        fi
        tmp=$(mktemp -d)
        trap 'rm -rf "$tmp"' EXIT
        curl -fsSL {CNI_URL!r} -o "$tmp/cni.tgz"
        sudo -n mkdir -p /opt/cni/bin /etc/cni/net.d
        sudo -n tar -C /opt/cni/bin -xzf "$tmp/cni.tgz"
        test -x /opt/cni/bin/bridge
        """
    ).strip()


def cni_write_conflist_script(mode: NetworkMode) -> str:
    if mode == NetworkMode.tsot:
        path = TSOT_CONFLIST
        body = tsot_conflist_body()
        remove_other = f"sudo -n rm -f {BRIDGE_CONFLIST}"
    else:
        path = BRIDGE_CONFLIST
        body = bridge_conflist_body()
        remove_other = f"sudo -n rm -f {TSOT_CONFLIST}"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        sudo -n mkdir -p /opt/cni/bin /etc/cni/net.d
        {remove_other}
        sudo -n tee {path} >/dev/null <<'JSON'
{body}
JSON
        {cni_conflist_type_script(mode.value)}
        """
    ).strip()


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


class CniPluginsStep(SetupStep):
    def __init__(self, mode: NetworkMode = NetworkMode.bridge):
        self.mode = mode
        self.name = f"cni-{mode.value}"

    def run(self, remote: RemoteHost) -> StepResult:
        r = remote.sh(cni_install_binaries_script(), timeout=300)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        r = remote.sh(cni_write_conflist_script(self.mode), timeout=60)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        if self.mode == NetworkMode.bridge:
            fwd = remote.sh(cni_forward_script(), timeout=60)
            if not fwd.ok:
                return StepResult(self.name, False, remote.format_failure(fwd))
        return StepResult(self.name, True, f"cni {self.mode.value} conflist ready")
