"""TSOT stack setup scripts (lab host only)."""

from __future__ import annotations

import shlex
import textwrap

from keska_lab.profile import NetworkParams
from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupStep, StepResult
from keska_lab.setup.quark_config import QUARK_CONFIG, bench_config_json, deploy_config_script

QSERVICE_BIN = "target/debug"
QLET_CONFIG_PATH = "/etc/quark/lab-qlet.json"
ETCD_NAME = "keska-etcd"
ETCD_IMAGE = "quay.io/coreos/etcd:v3.5.16"
SS_NAME = "keska-ss"
CADVISOR_NAME = "keska-cadvisor"
CADVISOR_IMAGE = "gcr.io/cadvisor/cadvisor:v0.36.0"
TSOT_SOCKET = "/var/run/quark/tsot-socket"
GRPCURL_VERSION = "1.9.1"
GRPCURL_URL = (
    f"https://github.com/fullstorydev/grpcurl/releases/download/"
    f"v{GRPCURL_VERSION}/grpcurl_{GRPCURL_VERSION}_linux_x86_64.tar.gz"
)


def deploy_qlet_config_script(params: NetworkParams) -> str:
    body = textwrap.dedent(
        f"""
        {{
          "nodeName": "node1",
          "etcdAddresses": ["127.0.0.1:2379"],
          "nodeIp": "{params.node_ip}",
          "podMgrPort": {params.pod_mgr_port},
          "tsotCniPort": {params.tsot_cni_port},
          "tsotSvcPort": {params.tsot_svc_port},
          "stateSvcPort": {params.qlet_state_svc_port},
          "cidr": "{params.cidr}",
          "stateSvcAddr": ["127.0.0.1:{params.state_svc_port}"],
          "singleNodeModel": true
        }}
        """
    ).strip()
    return textwrap.dedent(
        f"""
        set -euo pipefail
        sudo -n tee {QLET_CONFIG_PATH} >/dev/null <<'JSON'
{body}
JSON
        echo "deployed {QLET_CONFIG_PATH}"
        """
    ).strip()


def cadvisor_start_script() -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        if sudo -n docker ps --format '{{{{.Names}}}}' | grep -qx {CADVISOR_NAME}; then
          echo "cadvisor already running"
          exit 0
        fi
        sudo -n docker rm -f {CADVISOR_NAME} >/dev/null 2>&1 || true
        sudo -n docker run -d --name {CADVISOR_NAME} \\
          --volume=/:/rootfs:ro \\
          --volume=/var/run:/var/run:ro \\
          --volume=/sys:/sys:ro \\
          --volume=/var/lib/docker/:/var/lib/docker:ro \\
          --volume=/dev/disk/:/dev/disk:ro \\
          --publish=8080:8080 \\
          --privileged \\
          --device=/dev/kmsg \\
          {CADVISOR_IMAGE}
        sleep 3
        sudo -n docker ps --filter name={CADVISOR_NAME} --filter status=running -q | grep -q .
        curl -sf http://127.0.0.1:8080/api/v2.1/machine >/dev/null
        """
    ).strip()


def etcd_start_script() -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        if sudo -n docker ps --format '{{{{.Names}}}}' | grep -qx {ETCD_NAME}; then
          echo "etcd already running"
          exit 0
        fi
        sudo -n docker rm -f {ETCD_NAME} >/dev/null 2>&1 || true
        sudo -n docker run -d --name {ETCD_NAME} \\
          -p 2379:2379 -p 2380:2380 \\
          {ETCD_IMAGE} \\
          /usr/local/bin/etcd \\
          --advertise-client-urls http://127.0.0.1:2379 \\
          --listen-client-urls http://0.0.0.0:2379 \\
          --listen-peer-urls http://0.0.0.0:2380 \\
          --initial-advertise-peer-urls http://127.0.0.1:2380 \\
          --initial-cluster default=http://127.0.0.1:2380
        sleep 3
        sudo -n docker ps --filter name={ETCD_NAME} --filter status=running -q | grep -q .
        """
    ).strip()


def ss_start_script(repo: str, port: int = 8890) -> str:
    bindir = f"$REPO/qservice/{QSERVICE_BIN}"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        REPO={shlex.quote(repo)}
        BINDIR={bindir}
        {etcd_ready_script()}
        if {IPROUTE_SS} -lntp 2>/dev/null | grep -q ':{port}'; then
          echo "ss already listening on {port}"
          exit 0
        fi
        if [ ! -x "$BINDIR/ss" ]; then
          echo "ss binary missing at $BINDIR/ss (build qservice first)" >&2
          exit 1
        fi
        {_stop_qservice_ss_script()}
        sleep 1
        sudo -n nohup "$BINDIR/ss" {QLET_CONFIG_PATH} >/dev/null 2>&1 &
        for i in $(seq 1 20); do
          if {IPROUTE_SS} -lntp 2>/dev/null | grep -q ':{port}'; then
            echo "ss ready on {port}"
            exit 0
          fi
          sleep 1
        done
        echo "ss not ready on {port}" >&2
        exit 1
        """
    ).strip()


def grpcurl_install_script() -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        if command -v grpcurl >/dev/null 2>&1; then
          echo "grpcurl present"
          exit 0
        fi
        tmp=$(mktemp -d)
        trap 'rm -rf "$tmp"' EXIT
        curl -fsSL {GRPCURL_URL!r} -o "$tmp/grpcurl.tgz"
        sudo -n tar -C /usr/local/bin -xzf "$tmp/grpcurl.tgz" grpcurl
        test -x /usr/local/bin/grpcurl
        """
    ).strip()


def qservice_build_script(repo: str) -> str:
    qdir = f"{shlex.quote(repo)}/qservice"
    lib = f"{qdir}/qshare/src/lib.rs"
    na_bin = f"{qdir}/{QSERVICE_BIN}/na"
    conn_src = f"{qdir}/qlet/tsot/conn_svc.rs"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        source "$HOME/.cargo/env" 2>/dev/null || true
        export PATH="$HOME/.cargo/bin:$PATH"
        command -v cargo >/dev/null || {{ echo "cargo not in PATH" >&2; exit 1; }}
        cd {qdir}
        if grep -q 'runtime.v1alpha2.rs' {lib}; then
          sed -i 's/runtime.v1alpha2.rs/runtime.v1.rs/' {lib}
        fi
        make na cni ss
        test -x {na_bin}
        test -x {qdir}/{QSERVICE_BIN}/cni
        test -x {qdir}/{QSERVICE_BIN}/ss
        if [ "$(stat -c %Y {na_bin})" -lt "$(stat -c %Y {conn_src})" ]; then
          echo "na binary older than conn_svc.rs — build did not pick up changes" >&2
          exit 1
        fi
        sudo -n cp -f {qdir}/{QSERVICE_BIN}/cni /opt/cni/bin/tsot
        echo "qservice built"
        """
    ).strip()


IPROUTE_SS = "/bin/ss"


def _stop_qservice_ss_script() -> str:
    return textwrap.dedent(
        """
        sudo -n pkill -x ss 2>/dev/null || true
        sudo -n pkill -f "qservice/target/.*/ss" 2>/dev/null || true
        sudo -n pkill -f "/etc/quark/lab-qlet.json" 2>/dev/null || true
        sudo -n pkill -f "/etc/quark/lab-ss.json" 2>/dev/null || true
        sudo -n rm -f {TSOT_SOCKET}
        """
    ).strip()


def etcd_ready_script() -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        for i in $(seq 1 15); do
          if sudo -n docker ps --format '{{{{.Names}}}}' | grep -qx {ETCD_NAME}; then
            if curl -sf http://127.0.0.1:2379/health >/dev/null 2>&1; then
              echo "etcd ready"
              exit 0
            fi
          fi
          sleep 1
        done
        echo "etcd not ready" >&2
        exit 1
        """
    ).strip()


def na_stop_script(repo: str) -> str:
    bindir = f"$REPO/qservice/{QSERVICE_BIN}"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        REPO={shlex.quote(repo)}
        BINDIR={bindir}
        sudo -n pkill -x na 2>/dev/null || true
        sudo -n pkill -f "$BINDIR/na" 2>/dev/null || true
        {_stop_qservice_ss_script()}
        sleep 2
        echo "na/ss stopped"
        """
    ).strip()


def qservice_start_script(repo: str) -> str:
    bindir = f"$REPO/qservice/{QSERVICE_BIN}"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        REPO={shlex.quote(repo)}
        BINDIR={bindir}
        {na_stop_script(repo)}
        sudo -n mkdir -p /var/log/quark /var/run/quark
        sudo -n touch /var/log/quark/na.log
        sudo -n chmod 644 /var/log/quark/na.log
        sudo -n nohup "$BINDIR/na" {QLET_CONFIG_PATH} >/dev/null 2>&1 &
        sleep 8
        {na_liveness_script()}
        """
    ).strip()


def na_liveness_script(port: int = 8888) -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        for i in $(seq 1 30); do
          if [ -S {TSOT_SOCKET} ] && pgrep -x na >/dev/null; then
            {IPROUTE_SS} -lntp 2>/dev/null | grep -q ':{port}' || continue
            echo "na ready"
            exit 0
          fi
          sleep 1
        done
        echo "na not ready" >&2
        pgrep -af na >&2 || true
        tail -20 /var/log/quark/na.log >&2 || true
        exit 1
        """
    ).strip()


def tsot_egress_host_script(cidr: str) -> str:
    """Host routes + SNAT so TSOT pod IPs can reach the internet."""
    cidr_q = shlex.quote(cidr)
    return textwrap.dedent(
        f"""
        set -euo pipefail
        CIDR=$(python3 - <<'PY'
import ipaddress
print(ipaddress.ip_network({cidr_q!r}, strict=False))
PY
)
        sudo -n sysctl -w net.ipv4.ip_forward=1 >/dev/null
        sudo -n ip route replace local "$CIDR" dev lo
        if ! sudo -n iptables -t nat -C POSTROUTING -s "$CIDR" ! -d "$CIDR" -j MASQUERADE 2>/dev/null; then
          sudo -n iptables -t nat -A POSTROUTING -s "$CIDR" ! -d "$CIDR" -j MASQUERADE
        fi
        echo "tsot egress host routing for $CIDR"
        """
    ).strip()


def tsot_stack_stop_script(repo: str) -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        {na_stop_script(repo)}
        sudo -n docker rm -f {ETCD_NAME} {SS_NAME} 2>/dev/null || true
        echo "tsot stack stopped"
        """
    ).strip()


class TsotStackStep(SetupStep):
    name = "tsot-stack"

    def __init__(self, profile: NodeProfile):
        self.profile = profile

    def run(self, remote: RemoteHost) -> StepResult:
        repo = remote.config.remote_repo
        params = self.profile.net_params
        steps = [
            deploy_config_script(bench_config_json(self.profile)),
            deploy_qlet_config_script(params),
            tsot_egress_host_script(params.cidr),
            grpcurl_install_script(),
            cadvisor_start_script(),
            etcd_start_script(),
            qservice_build_script(repo),
            ss_start_script(repo, port=params.state_svc_port),
            qservice_start_script(repo),
        ]
        for script in steps:
            r = remote.sh(script, timeout=900)
            if not r.ok:
                return StepResult(self.name, False, remote.format_failure(r))
        return StepResult(self.name, True, "tsot stack ready")


class TsotStackStopStep(SetupStep):
    name = "tsot-stack-stop"

    def run(self, remote: RemoteHost) -> StepResult:
        r = remote.sh(tsot_stack_stop_script(remote.config.remote_repo), timeout=120)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        return StepResult(self.name, True, "tsot stack stopped")
