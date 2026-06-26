"""Deploy /etc/quark/config.json for lab benchmarks (host only)."""

from __future__ import annotations

import json
import textwrap

from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupStep, StepResult

QUARK_CONFIG = "/etc/quark/config.json"

# Mirrors repo config.json — keep in sync when Quark adds required fields.
DEFAULT_QUARK_CONFIG: dict = {
    "DebugLevel": "Error",
    "KernelMemSize": 24,
    "LogType": "Sync",
    "LogLevel": "Simple",
    "CudaMemType": "Default",
    "UringIO": True,
    "UringFixedFile": False,
    "EnableAIO": True,
    "PrintException": False,
    "KernelPagetable": False,
    "PerfDebug": False,
    "UringStatx": False,
    "FileBufWrite": True,
    "MmapRead": True,
    "AsyncAccept": True,
    "EnableRDMA": False,
    "RDMAPort": 1,
    "PerSandboxLog": False,
    "ReserveCpuCount": 1,
    "ShimMode": False,
    "EnableInotify": True,
    "ReaddirCache": True,
    "HiberODirect": True,
    "DisableCgroup": True,
    "CopyDataWithPf": True,
    "TlbShootdownWait": True,
    "Sandboxed": False,
    "Realtime": False,
    "EnableIOBuf": True,
    "EnableTsot": False,
    "CCMode": "None",
}


def bench_config_json(*, enable_tsot: bool = False) -> dict:
    cfg = dict(DEFAULT_QUARK_CONFIG)
    if enable_tsot:
        cfg["EnableTsot"] = True
        cfg["ShimMode"] = True
        cfg["PerSandboxLog"] = True
    return cfg


def deploy_config_script(cfg: dict) -> str:
    body = json.dumps(cfg, indent=2)
    return textwrap.dedent(
        f"""
        set -euo pipefail
        sudo -n mkdir -p /etc/quark /var/run/quark /var/log/quark
        sudo -n tee {QUARK_CONFIG} >/dev/null <<'JSON'
{body}
JSON
        echo "deployed {QUARK_CONFIG}"
        """
    ).strip()


class QuarkBenchConfigStep(SetupStep):
    """Ensure default (non-TSOT) Quark config for standard benchmarks."""

    name = "quark-bench-config"

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        r = remote.sh(deploy_config_script(bench_config_json()), timeout=60, stream=stream)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        return StepResult(self.name, True, "fast IO bench config (UringIO, MmapRead, EnableIOBuf)")
