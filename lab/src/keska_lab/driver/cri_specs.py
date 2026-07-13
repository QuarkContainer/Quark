from __future__ import annotations


TSOT_POD_DNS = {"servers": ["127.0.0.53"]}
BRIDGE_POD_DNS = {"servers": ["8.8.8.8", "1.1.1.1"]}


def pod_spec(
    *,
    name: str,
    uid: str,
    runtime_handler: str = "",
    tsot_dns: bool = False,
) -> dict:
    pod: dict = {
        "metadata": {"name": name, "uid": uid, "namespace": "default"},
        "log_directory": f"/tmp/keska-cri-{uid}",
        "linux": {},
        "dns_config": TSOT_POD_DNS if tsot_dns else BRIDGE_POD_DNS,
    }
    if runtime_handler:
        pod["runtime_handler"] = runtime_handler
    return pod

