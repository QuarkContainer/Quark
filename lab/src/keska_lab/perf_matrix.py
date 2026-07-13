"""Run a network perf matrix across NodeProfiles (driver-based).

This replaces the old shell-harness matrix runner: it delegates to the colocated driver
via `keska-lab-node driver ...` style operations, but is callable directly as a library
from `node_cli.py`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from keska_lab.config import LabConfig
from keska_lab.driver.protocol import DriverEnvelope
from keska_lab.installer import InstallOptions
from keska_lab.knode import install_node
from keska_lab.profile import NodeProfile
from keska_lab.remote import RemoteHost


MATRIX_PROFILES = ("quark_tsot", "quark_bridge", "kata_bridge")


@dataclass
class MatrixRow:
    profile: str
    install_ok: bool = True
    bench_ok: bool = True
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    note: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _results_path(cfg: LabConfig) -> Path:
    root = Path.home() / ".keska-lab" / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"matrix_{_now_iso().replace(':', '_')}.json"


def _wait_for_node_ready(remote: RemoteHost, profile: NodeProfile, *, timeout_s: int = 180) -> None:
    from keska_lab.installer import NodeInstaller

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rep = NodeInstaller(remote, profile).verify(network_only=True)
        if rep.ok:
            return
        time.sleep(2)
    raise TimeoutError(f"node not ready after {timeout_s}s for {profile.name}")


def _run_driver_op(remote: RemoteHost, cfg: LabConfig, *, op: str, input_obj: dict, request_id: str) -> DriverEnvelope:
    venv_python = f"{cfg.remote_repo}/lab/.venv/bin/python"
    # Write input json in a temp dir on the host and invoke driver.
    payload = json.dumps(input_obj, indent=2)
    cmd = (
        "set -euo pipefail; "
        "tmp=/tmp/keska-matrix-$RANDOM; "
        "mkdir -p \"$tmp\"; "
        "cat >\"$tmp/in.json\" <<'JSON'\n"
        f"{payload}\n"
        "JSON\n"
        f"{venv_python} -m keska_lab.driver run --protocol 1 --request-id {request_id} "
        f"--op {op} --input \"$tmp/in.json\""
    )
    r = remote.run(cmd, timeout=420, check=False)
    if not r.ok:
        raise RuntimeError(f"driver op failed: op={op} rc={r.returncode} stderr={r.stderr.strip()}")
    return DriverEnvelope.model_validate(json.loads(r.stdout))


def run_perf_matrix(
    cfg: LabConfig,
    *,
    skip_install: bool = False,
    restore_profile: str = "quark_tsot",
) -> list[MatrixRow]:
    remote = RemoteHost(cfg)
    rows: list[MatrixRow] = []

    def prof(name: str) -> NodeProfile:
        if name == "quark_tsot":
            return NodeProfile.quark_tsot()
        if name == "quark_bridge":
            return NodeProfile.quark_bridge()
        if name == "kata_bridge":
            return NodeProfile.kata_bridge()
        raise ValueError(name)

    for p in MATRIX_PROFILES:
        row = MatrixRow(profile=p)
        try:
            profile = prof(p)
            if not skip_install:
                install_node(
                    cfg,
                    profile,
                    options=InstallOptions(
                        gate_level="L1",
                        network_only=False,
                        skip_provision=True,
                        skip_gates=False,
                        skip_containerd=False,
                    ),
                )
            _wait_for_node_ready(remote, profile, timeout_s=180)

            # Driver network ops: these are stable even when iperf is platform-blocked.
            env1 = _run_driver_op(
                remote,
                cfg,
                op="network.inet_connect",
                input_obj={
                    "image": "python:3.12-slim",
                    "runtime_handler": "kata" if profile.runtime == "kata" else "quark",
                    "tsot_dns": profile.network.value == "tsot",
                },
                request_id=f"matrix-{p}-inet_connect",
            )
            if env1.status.value != "ok":
                raise RuntimeError(f"inet_connect failed: {env1.to_json()}")
            row.metrics["inet_tcp_connect_ms"] = float(env1.result["connect_ms"])

            env2 = _run_driver_op(
                remote,
                cfg,
                op="network.inet_download",
                input_obj={
                    "image": "python:3.12-slim",
                    "runtime_handler": "kata" if profile.runtime == "kata" else "quark",
                    "tsot_dns": profile.network.value == "tsot",
                },
                request_id=f"matrix-{p}-inet_download",
            )
            if env2.status.value != "ok":
                raise RuntimeError(f"inet_download failed: {env2.to_json()}")
            row.metrics["inet_download_mbps"] = float(env2.result["mbps"])

        except Exception as e:
            row.bench_ok = False
            row.errors.append(str(e))
        rows.append(row)

    # Restore default profile at end (best-effort).
    try:
        if restore_profile in MATRIX_PROFILES and not skip_install:
            install_node(cfg, prof(restore_profile), options=InstallOptions(gate_level="L1", skip_provision=True))
    except Exception:
        pass

    _results_path(cfg).write_text(json.dumps([r.__dict__ for r in rows], indent=2) + "\n")
    return rows

