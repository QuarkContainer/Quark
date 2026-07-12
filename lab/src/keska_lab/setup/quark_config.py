"""Deploy /etc/quark/config.json for lab benchmarks (host only)."""

from __future__ import annotations

import json
import textwrap

from keska_lab.profile import NetworkMode, NodeProfile
from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupStep, StepResult

QUARK_CONFIG = "/etc/quark/config.json"

DEFAULT_QUARK_CONFIG: dict = {
    "DebugLevel": "Error",
    "KernelMemSize": 24,
    "LogType": "Sync",
    "LogLevel": "Simple",
    "CudaMemType": "Default",
    "UringIO": True,
    "EnableAIO": True,
    "PrintException": False,
    "KernelPagetable": False,
    "PerfDebug": False,
    "UringStatx": False,
    "MmapRead": False,
    "AsyncAccept": True,
    "EnableRDMA": False,
    "RDMAPort": 1,
    "PerSandboxLog": False,
    "ReserveCpuCount": 1,
    "EnableInotify": True,
    "ReaddirCache": True,
    "HiberODirect": True,
    "DisableCgroup": False,
    "CopyDataWithPf": True,
    "TlbShootdownWait": True,
    "Sandboxed": False,
    "Realtime": False,
    "EnableTsot": False,
    "CCMode": "None",
}


def bench_config_json(profile: NodeProfile | None = None, *, enable_tsot: bool | None = None) -> dict:
    """Build Quark config from NodeProfile (preferred) or legacy enable_tsot flag."""
    if profile is None and enable_tsot is None:
        profile = NodeProfile.quark_bridge()
    cfg = dict(DEFAULT_QUARK_CONFIG)
    if profile is not None:
        cfg["EnableTsot"] = profile.network == NetworkMode.tsot
        cfg["EnableRDMA"] = profile.network == NetworkMode.rdma
        if profile.network == NetworkMode.tsot:
            cfg["PerSandboxLog"] = True
    elif enable_tsot:
        cfg["EnableTsot"] = True
        cfg["PerSandboxLog"] = True
    return cfg


def cri_bench_config_json() -> dict:
    cfg = bench_config_json(NodeProfile.quark_bridge())
    cfg["Sandboxed"] = True
    cfg["DisableCgroup"] = False
    return cfg


EXPERIMENTAL_FLAG_KEYS = ("MmapRead", "UringStatx")
GROUP5_ITERATIONS = 5
GROUP5_SUITES = ("light", "full", "db")
GROUP5_BENCH_MATRIX: dict[str, tuple[str, ...]] = {
    "MmapRead": GROUP5_SUITES,
    "UringStatx": GROUP5_SUITES,
}


def experimental_config_json(flag: str) -> dict:
    if flag not in EXPERIMENTAL_FLAG_KEYS:
        raise ValueError(f"unknown experimental flag {flag!r}; expected one of {EXPERIMENTAL_FLAG_KEYS}")
    cfg = bench_config_json(NodeProfile.quark_bridge())
    for key in EXPERIMENTAL_FLAG_KEYS:
        cfg[key] = key == flag
    return cfg


def experimental_cargo_features(flag: str) -> list[str]:
    mapping = {
        "MmapRead": ["experimental-mmap-read"],
        "UringStatx": ["experimental-uring-statx"],
    }
    return mapping[flag]


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


def quark_config_enable_tsot_script(expected: bool) -> str:
    expected_py = "True" if expected else "False"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        python3 -c "import json; c=json.load(open('{QUARK_CONFIG}')); exit(0 if c.get('EnableTsot') is {expected_py} else 1)"
        echo ok
        """
    ).strip()


class QuarkConfigStep(SetupStep):
    """Deploy Quark config derived from NodeProfile."""

    def __init__(self, profile: NodeProfile):
        self.profile = profile
        self.name = "quark-config"

    def run(self, remote: RemoteHost) -> StepResult:
        if self.profile.runtime != "quark":
            return StepResult(self.name, True, "skip (kata runtime)")
        cfg = bench_config_json(self.profile)
        r = remote.sh(deploy_config_script(cfg), timeout=60)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        r = remote.sh(quark_config_enable_tsot_script(self.profile.network == NetworkMode.tsot), timeout=30)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        return StepResult(
            self.name,
            True,
            f"EnableTsot={self.profile.network == NetworkMode.tsot}",
        )


class QuarkBenchConfigStep(SetupStep):
    """Legacy: default bridge Quark config."""

    name = "quark-bench-config"

    def run(self, remote: RemoteHost) -> StepResult:
        return QuarkConfigStep(NodeProfile.quark_bridge()).run(remote)


class QuarkExperimentalConfigStep(SetupStep):
    name = "quark-experimental-config"

    def __init__(self, flag: str) -> None:
        self.flag = flag

    def run(self, remote: RemoteHost) -> StepResult:
        cfg = experimental_config_json(self.flag)
        features = " ".join(experimental_cargo_features(self.flag))
        r = remote.sh(deploy_config_script(cfg), timeout=60)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        return StepResult(
            self.name,
            True,
            f"experimental config {self.flag}=true (build with: CARGO_FEATURES={features})",
        )
