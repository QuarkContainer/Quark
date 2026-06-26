"""TSOT stack setup for Quark network benchmarks (lab host only)."""

from __future__ import annotations

import shlex
import textwrap

from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupStep, StepResult
from keska_lab.setup.quark_config import QUARK_CONFIG, bench_config_json, deploy_config_script

QSERVICE_BIN = "target/debug"
QLET_CONFIG_PATH = "/etc/quark/lab-qlet.json"
ETCD_NAME = "keska-etcd"
ETCD_IMAGE = "quay.io/coreos/etcd:v3.5.16"
CADVISOR_NAME = "keska-cadvisor"
CADVISOR_IMAGE = "gcr.io/cadvisor/cadvisor:v0.36.0"
TSOT_SOCKET = "/var/run/quark/tsot-socket"


def tsot_config_json() -> dict:
    return bench_config_json(enable_tsot=True)


def deploy_tsot_config_script() -> str:
    return deploy_config_script(tsot_config_json())


def deploy_qlet_config_script() -> str:
    """Lab qlet config — single-node, ports aligned with ss (8890) and etcd (2379)."""
    body = textwrap.dedent(
        """
        {
          "nodeName": "node1",
          "etcdAddresses": ["127.0.0.1:2379"],
          "nodeIp": "127.0.0.1",
          "podMgrPort": 8888,
          "tsotCniPort": 1234,
          "tsotSvcPort": 1235,
          "stateSvcPort": 8890,
          "cidr": "10.1.1.0/8",
          "stateSvcAddr": ["127.0.0.1:8890"],
          "singleNodeModel": true
        }
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


def qservice_build_script(repo: str) -> str:
    qdir = f"{shlex.quote(repo)}/qservice"
    lib = f"{qdir}/qshare/src/lib.rs"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        export PATH="$HOME/.cargo/bin:$PATH"
        cd {qdir}
        if grep -q 'runtime.v1alpha2.rs' {lib}; then
          sed -i 's/runtime.v1alpha2.rs/runtime.v1.rs/' {lib}
        fi
        if [ ! -x {QSERVICE_BIN}/na ] || [ ! -x {QSERVICE_BIN}/cni ]; then
          make na cni
        fi
        test -x {QSERVICE_BIN}/na
        test -x {QSERVICE_BIN}/cni
        sudo -n cp -f {QSERVICE_BIN}/cni /opt/cni/bin/tsot
        echo "qservice built (cri v1)"
        """
    ).strip()


def qservice_start_script(repo: str) -> str:
    bindir = f"$REPO/qservice/{QSERVICE_BIN}"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        REPO={shlex.quote(repo)}
        BINDIR={bindir}
        sudo -n pkill -x na 2>/dev/null || true
        sudo -n pkill -f "$BINDIR/na" 2>/dev/null || true
        sudo -n pkill -f "$BINDIR/ss" 2>/dev/null || true
        sleep 2
        sudo -n mkdir -p /var/log/quark /var/run/quark
        sudo -n touch /var/log/quark/na.log
        sudo -n chmod 644 /var/log/quark/na.log
        sudo -n nohup "$BINDIR/na" {QLET_CONFIG_PATH} >/dev/null 2>&1 &
        sleep 8
        for i in $(seq 1 30); do
          if [ -S {TSOT_SOCKET} ] && pgrep -f "$BINDIR/na" >/dev/null; then
            ss -lntp | grep -q ':8888' || continue
            echo "tsot stack ready"
            exit 0
          fi
          sleep 1
        done
        echo "TSOT stack not ready after 30s" >&2
        pgrep -af "$BINDIR" >&2 || true
        tail -20 /var/log/quark/na.log >&2 || true
        exit 1
        """
    ).strip()


def tsot_ready_script() -> str:
    return (
        f"test -S {TSOT_SOCKET} && "
        f"python3 -c \"import json; c=json.load(open('{QUARK_CONFIG}')); "
        f"exit(0 if c.get('EnableTsot') else 1)\" && "
        f"pgrep -f 'qservice/target/debug/na' >/dev/null"
    )


def ensure_tsot_stack(remote: RemoteHost, *, stream: bool = False) -> str:
    repo = remote.config.remote_repo
    steps = [
        deploy_tsot_config_script(),
        deploy_qlet_config_script(),
        cadvisor_start_script(),
        etcd_start_script(),
        qservice_build_script(repo),
        qservice_start_script(repo),
    ]
    for script in steps:
        r = remote.sh(script, timeout=900, stream=stream)
        if not r.ok:
            raise RuntimeError(remote.format_failure(r))
    return "tsot stack ready"


class TsotBenchReadyStep(SetupStep):
    name = "tsot-stack"

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        try:
            msg = ensure_tsot_stack(remote, stream=stream)
            return StepResult(self.name, True, msg)
        except Exception as e:
            return StepResult(self.name, False, str(e))
