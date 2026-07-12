"""TSOT gate scripts — CreatePod, CNI lookup, optional crictl lifecycle."""

from __future__ import annotations

import json
import shlex
import textwrap
import uuid


def minimal_pod_def(uid: str) -> dict:
    return {
        "tenant": "lab",
        "namespace": "default",
        "funcname": "gate",
        "id": "1",
        "uid": uid,
        "resource_version": "",
        "labels": {},
        "annotations": {},
        "init_containers": [],
        "containers": [],
        "volumes": [],
        "host_network": False,
        "host_name": "",
        "host_ipc": False,
        "host_pid": False,
        "share_process_namespace": False,
        "overhead": {},
        "deletion_timestamp": None,
        "deletion_grace_period_seconds": None,
        "termination_grace_period_seconds": None,
        "runtime_class_name": None,
        "security_context": None,
        "ipAddr": 0,
        "status": {
            "host_ip": "",
            "pod_ip": "",
            "pod_ips": [],
            "phase": "Pending",
            "conditions": [],
            "start_time": None,
        },
        "state": "Init",
    }


def minimal_pod_def_json(uid: str) -> str:
    return json.dumps(minimal_pod_def(uid))


def _grpcurl_check_error(out_var: str = "$OUT") -> str:
    return f'echo {out_var} | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if not d.get(\\"error\\") else 1)"'


def tsot_create_pod_grpc_script(
    repo: str,
    uid: str,
    *,
    pod_mgr_port: int = 8888,
) -> str:
    pod_body = json.dumps(minimal_pod_def(uid))
    req_json = json.dumps({"pod": pod_body, "configMap": "{}"})
    proto_dir = f"{repo}/qservice/qshare/proto"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        command -v grpcurl >/dev/null
        OUT=$(grpcurl -plaintext \\
          -import-path {proto_dir} \\
          -proto na.proto \\
          -d {shlex.quote(req_json)} \\
          127.0.0.1:{pod_mgr_port} na.NodeAgentService/CreatePod)
        {_grpcurl_check_error()}
        echo "CreatePod ok uid={uid}"
        """
    ).strip()


def tsot_get_sandbox_addr_script(
    repo: str,
    uid: str,
    *,
    tsot_cni_port: int = 1234,
    namespace: str = "lab/default",
) -> str:
    proto_dir = f"{repo}/qservice/qshare/proto"
    req = json.dumps(
        {
            "pod_uid": uid,
            "namespace": namespace,
            "pod_name": "gate_1",
            "container_id": "",
        }
    )
    return textwrap.dedent(
        f"""
        set -euo pipefail
        command -v grpcurl >/dev/null
        OUT=$(grpcurl -plaintext \\
          -import-path {proto_dir} \\
          -proto tsot_cni.proto \\
          -d {json.dumps(req)} \\
          127.0.0.1:{tsot_cni_port} tsot_cni.TsotCniService/GetPodSandboxAddr)
        echo "$OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); v=d.get('ipAddr',d.get('ip_addr',0)); exit(0 if int(v)>0 else 1)"
        echo "GetPodSandboxAddr ok"
        """
    ).strip()


def tsot_remove_sandbox_script(
    repo: str,
    uid: str,
    *,
    tsot_cni_port: int = 1234,
    namespace: str = "lab/default",
) -> str:
    proto_dir = f"{repo}/qservice/qshare/proto"
    req = json.dumps(
        {
            "pod_uid": uid,
            "namespace": namespace,
            "pod_name": "gate_1",
            "container_id": "",
        }
    )
    return textwrap.dedent(
        f"""
        set -euo pipefail
        command -v grpcurl >/dev/null
        OUT=$(grpcurl -plaintext \\
          -import-path {proto_dir} \\
          -proto tsot_cni.proto \\
          -d {json.dumps(req)} \\
          127.0.0.1:{tsot_cni_port} tsot_cni.TsotCniService/RemovePodSandbox) || true
        echo "RemovePodSandbox ok"
        """
    ).strip()


def tsot_gate_l1_script(repo: str, *, pod_mgr_port: int = 8888, tsot_cni_port: int = 1234) -> str:
    uid = new_gate_uid()
    return textwrap.dedent(
        f"""
        set -euo pipefail
        GATE_POD_UID={uid}
        cleanup() {{
          set +e
          {tsot_remove_sandbox_script(repo, uid, tsot_cni_port=tsot_cni_port)} >/dev/null 2>&1
        }}
        trap cleanup EXIT
        {tsot_create_pod_grpc_script(repo, uid, pod_mgr_port=pod_mgr_port)}
        {tsot_get_sandbox_addr_script(repo, uid, tsot_cni_port=tsot_cni_port)}
        echo "TSOT L1 PASS"
        """
    ).strip()


def tsot_register_uid_script(
    repo: str,
    uid: str,
    *,
    pod_mgr_port: int = 8888,
) -> str:
    return tsot_create_pod_grpc_script(repo, uid, pod_mgr_port=pod_mgr_port)


def new_gate_uid() -> str:
    return str(uuid.uuid4())
