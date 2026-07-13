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


def inet_download_py(host_ip: str, host_header: str, path: str) -> str:
    del host_ip, host_header, path
    return """
import time, urllib.request
url = "http://speedtest.tele2.net/10MB.zip"
req = urllib.request.Request(url, headers={"User-Agent": "keska-lab"})
t0 = time.perf_counter()
data = urllib.request.urlopen(req, timeout=60).read()
elapsed = time.perf_counter() - t0
print(len(data) * 8 / elapsed / 1e6)
"""


INET_DOWNLOAD_HOST = "speedtest.tele2.net"
INET_DOWNLOAD_PATH = "/10MB.zip"
INET_DOWNLOAD_PY = inet_download_py("", INET_DOWNLOAD_HOST, INET_DOWNLOAD_PATH)

TSOT_POD_DNS = {"servers": ["127.0.0.53"]}
BRIDGE_POD_DNS = {"servers": ["8.8.8.8", "1.1.1.1"]}


def _cri_resolv_conf_script(*, tsot_dns: bool, cid_var: str = "CID") -> str:
    """TSOT guests need resolv.conf pointed at na's DNS proxy; bridge uses pod dns_config."""
    if not tsot_dns:
        return ""
    body = "\\n".join(f"nameserver {s}" for s in TSOT_POD_DNS["servers"])
    return (
        f'sudo -n crictl exec "${cid_var}" sh -c '
        f'"printf \\"{body}\\\\n\\" > /etc/resolv.conf" 2>/dev/null || true'
    )


def _ensure_cri_image(image: str, registry: str | None) -> str:
    """Shell prep: ctr pull (mirror-first when registry set) so crictl finds the canonical ref."""
    return ctr_pull_with_mirror_script(image, registry)


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


def _wait_sandbox_ip_script(
    ip_var: str = "IP",
    pod_var: str = "POD",
    *,
    timeout_s: int = 30,
) -> str:
    inspect = _sandbox_ip_from_inspect(ip_var, pod_var)
    return textwrap.dedent(
        f"""
        for _keska_ip_wait in $(seq 1 {timeout_s * 2}); do
          {inspect}
          if [ -n "${{{ip_var}}}" ]; then
            break
          fi
          sleep 0.5
        done
        test -n "${{{ip_var}}}" || {{
          echo "sandbox ${{{pod_var}}} has no pod ip after {timeout_s}s" >&2
          exit 1
        }}
        """
    ).strip()


def _wait_cri_exec_ready_script(cid_var: str = "CID", *, timeout_s: int = 30) -> str:
    return textwrap.dedent(
        f"""
        for _keska_exec_wait in $(seq 1 {timeout_s * 2}); do
          if sudo -n crictl exec "${{{cid_var}}}" true 2>/dev/null; then
            break
          fi
          sleep 0.5
        done
        sudo -n crictl exec "${{{cid_var}}}" true || {{
          echo "container ${{{cid_var}}} not exec-ready after {timeout_s}s" >&2
          exit 1
        }}
        """
    ).strip()


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
    tsot_dns: bool = False,
) -> dict:
    pod: dict = {
        "metadata": {"name": name, "uid": uid, "namespace": "default"},
        "log_directory": "/tmp",
        "linux": {},
        "dns_config": TSOT_POD_DNS if tsot_dns else BRIDGE_POD_DNS,
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
    tsot_dns: bool = False,
) -> str:
    import json
    import uuid as uuid_mod

    del runtime  # default_runtime_name=quark unless runtime_handler overrides
    pod_name = "keska-net"
    pod = _pod_spec(
        name=pod_name,
        uid=str(uuid_mod.uuid4()),
        runtime_handler=runtime_handler,
        tsot_dns=tsot_dns,
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
    resolv = _cri_resolv_conf_script(tsot_dns=tsot_dns)
    return textwrap.dedent(
        f"""
        set -euo pipefail
        WD=/tmp/keska-cri-$RANDOM
        sudo -n mkdir -p "$WD"
        sudo -n tee "$WD/pod.json" >/dev/null <<'JSON'
{pod_json}
JSON
        sudo -n tee "$WD/container.json" >/dev/null <<'JSON'
{container_json}
JSON
        {_ensure_cri_image(image, image_registry)}
        CID=$(sudo -n crictl run "$WD/container.json" "$WD/pod.json")
        sleep 2
        {resolv}
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
    tsot_dns: bool = False,
) -> str:
    import json
    import uuid as uuid_mod

    del runtime
    img = cri_image_ref(image)
    pod_s = _pod_spec(
        name="keska-iperf-s",
        uid=str(uuid_mod.uuid4()),
        runtime_handler=runtime_handler,
        tsot_dns=tsot_dns,
    )
    pod_c = _pod_spec(
        name="keska-iperf-c",
        uid=str(uuid_mod.uuid4()),
        runtime_handler=runtime_handler,
        tsot_dns=tsot_dns,
    )
    ctr_s = {
        "metadata": {"name": "iperf-s"},
        "image": {"image": img},
        "command": ["iperf3", "-s", "-p", "5201", "-1", "-B", "0.0.0.0"],
        "log_path": "iperf-s.log",
    }
    ctr_c = {
        "metadata": {"name": "iperf-c"},
        "image": {"image": img},
        "command": ["/bin/sleep", "600"],
        "log_path": "iperf-c.log",
    }
    resolv_c = _cri_resolv_conf_script(tsot_dns=tsot_dns, cid_var="CID_C")
    return textwrap.dedent(
        f"""
        set -euo pipefail
        WD=/tmp/keska-iperf-$RANDOM
        sudo -n mkdir -p "$WD"
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
        CID_S=$(sudo -n crictl run "$WD/ctr-s.json" "$WD/pod-s.json")
        {_sandbox_id_from_cid("POD_S", "CID_S")}
        {_wait_sandbox_ip_script("IP", "POD_S")}
        CID_C=$(sudo -n crictl run "$WD/ctr-c.json" "$WD/pod-c.json")
        {_wait_cri_exec_ready_script("CID_C")}
        {resolv_c}
        sleep 5
        sudo -n crictl exec "$CID_C" iperf3 -c "$IP" -p 5201 -t 5 -f m >/dev/null 2>&1 &
        _keska_iperf_client=$!
        out=""
        for _keska_iperf_poll in $(seq 1 20); do
          out=$(sudo -n crictl logs "$CID_S" 2>&1 || true)
          if echo "$out" | grep -qi Mbits; then
            break
          fi
          sleep 1
        done
        kill "$_keska_iperf_client" 2>/dev/null || true
        if ! echo "$out" | grep -qi Mbits; then
          echo "iperf produced no throughput in server logs" >&2
          exit 1
        fi
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
