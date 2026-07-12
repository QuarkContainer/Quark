"""Network benchmark helpers (TSOT-oriented, lab-side only)."""

from __future__ import annotations

import base64
import re
import shlex
import textwrap

from keska_lab.setup.image_registry import (
    DEFAULT_IMAGE_REGISTRY,
    cri_image_ref,
    ctr_pull_with_mirror_script,
)


def parse_iperf_mbps(output: str) -> float | None:
    for line in output.splitlines():
        m = re.search(r"([\d.]+)\s+Mbits/sec", line)
        if not m:
            continue
        if "receiver" in line.lower():
            return float(m.group(1))
    for line in output.splitlines():
        m = re.search(r"([\d.]+)\s+Mbits/sec", line)
        if m:
            return float(m.group(1))
    return None


def parse_dd_mib_s(line: str) -> float:
    m = re.search(r"([\d.]+)\s*([MG])B/s", line, re.I)
    if not m:
        return 0.0
    val = float(m.group(1))
    if m.group(2).upper() == "G":
        val *= 1024
    return round(val, 2)


def python_exec_cmd(code: str) -> str:
    encoded = base64.b64encode(code.encode()).decode()
    return (
        f"python3 -c \"import base64; exec(base64.b64decode('{encoded}').decode())\""
    )


INET_CONNECT_PY = """
import socket, time
host, port = "1.1.1.1", 443
t0 = time.perf_counter()
s = socket.create_connection((host, port), timeout=15)
s.close()
print(int((time.perf_counter() - t0) * 1000))
"""

INET_DOWNLOAD_PY = """
import time, urllib.request
url = "http://speedtest.tele2.net/10MB.zip"
req = urllib.request.Request(url, headers={"User-Agent": "keska-lab"})
t0 = time.perf_counter()
data = urllib.request.urlopen(req, timeout=60).read()
elapsed = time.perf_counter() - t0
print(len(data) * 8 / elapsed / 1e6)
"""

POD_DNS = {"servers": ["8.8.8.8", "1.1.1.1"]}


def _ensure_cri_image(image: str, registry: str | None) -> str:
    """Shell prep: ctr pull (mirror-first) so crictl finds the canonical ref."""
    return ctr_pull_with_mirror_script(image, registry or DEFAULT_IMAGE_REGISTRY)


def _sandbox_id_from_cid(var: str = "POD", cid_var: str = "CID") -> str:
    return (
        f'{var}=$(sudo -n crictl inspect "${cid_var}" 2>/dev/null '
        f"| python3 -c \"import sys,json; print(json.load(sys.stdin)['info']['sandboxID'])\")"
    )


def _sandbox_ip_from_inspect(var: str = "IP", pod_var: str = "POD") -> str:
    return (
        f'{var}=$(sudo -n crictl inspectp "${pod_var}" 2>/dev/null | python3 -c "'
        "import sys, json\n"
        "d = json.load(sys.stdin)\n"
        "for iface in d.get('info', {}).get('cniResult', {}).get('Interfaces', {}).values():\n"
        "    for cfg in iface.get('IPConfigs') or []:\n"
        "        ip = cfg.get('IP', '')\n"
        "        if ip and not ip.startswith('127.'):\n"
        "            print(ip)\n"
        "            raise SystemExit\n"
        '")'
    )


def crictl_config_script() -> str:
    return textwrap.dedent(
        """
        set -euo pipefail
        sudo -n tee /etc/crictl.yaml >/dev/null <<'EOF'
runtime-endpoint: unix:///run/containerd/containerd.sock
image-endpoint: unix:///run/containerd/containerd.sock
EOF
        """
    ).strip()


def _pod_spec(
    *,
    name: str,
    uid: str,
    runtime_handler: str = "",
) -> dict:
    pod: dict = {
        "metadata": {"name": name, "uid": uid, "namespace": "default"},
        "log_directory": "/tmp",
        "linux": {},
        "dns_config": POD_DNS,
    }
    if runtime_handler:
        pod["runtime_handler"] = runtime_handler
    return pod


def crictl_python_exec_script(
    code: str,
    *,
    image: str,
    runtime: str = "quark",
    runtime_handler: str = "",
    idle_cmd: list[str] | None = None,
    image_registry: str | None = DEFAULT_IMAGE_REGISTRY,
    tsot_repo: str | None = None,
    pod_mgr_port: int = 8888,
) -> str:
    import json
    import uuid as uuid_mod

    del runtime  # default_runtime_name=quark unless runtime_handler overrides
    pod_name = "keska-net"
    pod = _pod_spec(
        name=pod_name,
        uid=str(uuid_mod.uuid4()),
        runtime_handler=runtime_handler,
    )
    container = {
        "metadata": {"name": "keska-c"},
        "image": {"image": cri_image_ref(image)},
        "command": idle_cmd or ["/bin/sleep", "600"],
        "log_path": "keska.log",
    }
    py = python_exec_cmd(code)
    pod_json = json.dumps(pod)
    container_json = json.dumps(container)
    prereg = ""
    if tsot_repo:
        from keska_lab.gate.tsot_gate import tsot_register_uid_script

        prereg = tsot_register_uid_script(tsot_repo, pod["metadata"]["uid"], pod_mgr_port=pod_mgr_port)
    return textwrap.dedent(
        f"""
        set -euo pipefail
        WD=/tmp/keska-cri-$RANDOM
        sudo -n mkdir -p "$WD"
        {prereg}
        sudo -n tee "$WD/pod.json" >/dev/null <<'JSON'
{pod_json}
JSON
        sudo -n tee "$WD/container.json" >/dev/null <<'JSON'
{container_json}
JSON
        {_ensure_cri_image(image, image_registry)}
        CID=$(sudo -n crictl run "$WD/container.json" "$WD/pod.json" 2>/dev/null)
        sleep 2
        val=$(sudo -n crictl exec "$CID" {py})
        {_sandbox_id_from_cid()}
        sudo -n crictl rm -f "$CID" 2>/dev/null || true
        sudo -n crictl stopp "$POD" 2>/dev/null || true
        sudo -n crictl rmp "$POD" 2>/dev/null || true
        sudo -n rm -rf "$WD"
        echo "$val" | tail -1
        """
    ).strip()


def crictl_iperf_script(
    *,
    image: str = "networkstatic/iperf3",
    runtime: str = "quark",
    runtime_handler: str = "",
    image_registry: str | None = DEFAULT_IMAGE_REGISTRY,
    tsot_repo: str | None = None,
    pod_mgr_port: int = 8888,
) -> str:
    import json
    import uuid as uuid_mod

    del runtime
    img = cri_image_ref(image)
    pod_s = _pod_spec(
        name="keska-iperf-s",
        uid=str(uuid_mod.uuid4()),
        runtime_handler=runtime_handler,
    )
    pod_c = _pod_spec(
        name="keska-iperf-c",
        uid=str(uuid_mod.uuid4()),
        runtime_handler=runtime_handler,
    )
    ctr_s = {
        "metadata": {"name": "iperf-s"},
        "image": {"image": img},
        "command": ["/bin/sh", "-c", "iperf3 -s & exec sleep 600"],
        "log_path": "iperf-s.log",
    }
    ctr_c = {
        "metadata": {"name": "iperf-c"},
        "image": {"image": img},
        "command": ["/bin/sleep", "600"],
        "log_path": "iperf-c.log",
    }
    prereg = ""
    if tsot_repo:
        from keska_lab.gate.tsot_gate import tsot_register_uid_script

        prereg = "\n".join(
            [
                tsot_register_uid_script(tsot_repo, pod_s["metadata"]["uid"], pod_mgr_port=pod_mgr_port),
                tsot_register_uid_script(tsot_repo, pod_c["metadata"]["uid"], pod_mgr_port=pod_mgr_port),
            ]
        )
    return textwrap.dedent(
        f"""
        set -euo pipefail
        WD=/tmp/keska-iperf-$RANDOM
        sudo -n mkdir -p "$WD"
        {prereg}
        sudo -n tee "$WD/pod-s.json" >/dev/null <<'JSON'
{json.dumps(pod_s)}
JSON
        sudo -n tee "$WD/pod-c.json" >/dev/null <<'JSON'
{json.dumps(pod_c)}
JSON
        sudo -n tee "$WD/ctr-s.json" >/dev/null <<'JSON'
{json.dumps(ctr_s)}
JSON
        sudo -n tee "$WD/ctr-c.json" >/dev/null <<'JSON'
{json.dumps(ctr_c)}
JSON
        {_ensure_cri_image(image, image_registry)}
        CID_S=$(sudo -n crictl run "$WD/ctr-s.json" "$WD/pod-s.json" 2>/dev/null)
        sleep 3
        {_sandbox_id_from_cid("POD_S", "CID_S")}
        {_sandbox_ip_from_inspect("IP", "POD_S")}
        test -n "$IP"
        CID_C=$(sudo -n crictl run "$WD/ctr-c.json" "$WD/pod-c.json" 2>/dev/null)
        sleep 2
        out=$(sudo -n crictl exec "$CID_C" iperf3 -c "$IP" -t 5 -f m 2>&1 || true)
        CID="$CID_S"
        {_sandbox_id_from_cid("POD_S", "CID_S")}
        CID="$CID_C"
        {_sandbox_id_from_cid("POD_C", "CID_C")}
        sudo -n crictl rm -f "$CID_S" "$CID_C" 2>/dev/null || true
        sudo -n crictl stopp "$POD_S" "$POD_C" 2>/dev/null || true
        sudo -n crictl rmp "$POD_S" "$POD_C" 2>/dev/null || true
        sudo -n rm -rf "$WD"
        echo "$out"
        """
    ).strip()


def quark_exec_python(quark_cmd_fn, sandbox_id: str, code: str) -> str:
    py = python_exec_cmd(code)
    return quark_cmd_fn(f'exec --user 0:0 {shlex.quote(sandbox_id)} -- {py}')


def sandbox_ipv4_script(quark_cmd_fn, sandbox_id: str) -> str:
    return textwrap.dedent(
        f"""
        {quark_cmd_fn(f'exec --user 0:0 {shlex.quote(sandbox_id)} -- ip -4 -o addr show scope global')} 2>/dev/null \\
          | awk '{{print $4}}' | head -1 | cut -d/ -f1
        """
    ).strip()
