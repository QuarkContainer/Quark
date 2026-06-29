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
    "ShimMode": False,
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


def bench_config_json(*, enable_tsot: bool = False) -> dict:
    cfg = dict(DEFAULT_QUARK_CONFIG)
    if enable_tsot:
        cfg["EnableTsot"] = True
        cfg["ShimMode"] = True
        cfg["PerSandboxLog"] = True
    return cfg


def cri_bench_config_json() -> dict:
    """Quark config for CRI pods (H2 stats, multi-container sandbox)."""
    cfg = bench_config_json()
    cfg["ShimMode"] = True
    cfg["Sandboxed"] = True
    cfg["DisableCgroup"] = False
    return cfg


# Group 5 (E2/E3): runtime config for lab A/B when the matching Cargo feature is enabled.
EXPERIMENTAL_FLAG_KEYS = ("MmapRead", "UringStatx")

# Unified lab matrix (see kdoc/aaa-runtime-roadmap.md Group 5).
GROUP5_ITERATIONS = 5
GROUP5_SUITES = ("light", "full", "db")

GROUP5_BENCH_MATRIX: dict[str, tuple[str, ...]] = {
    "MmapRead": GROUP5_SUITES,
    "UringStatx": GROUP5_SUITES,
}


def experimental_config_json(flag: str) -> dict:
    """Return production bench config with one experimental I/O flag enabled."""
    if flag not in EXPERIMENTAL_FLAG_KEYS:
        raise ValueError(f"unknown experimental flag {flag!r}; expected one of {EXPERIMENTAL_FLAG_KEYS}")
    cfg = bench_config_json()
    for key in EXPERIMENTAL_FLAG_KEYS:
        cfg[key] = key == flag
    return cfg


def experimental_cargo_features(flag: str) -> list[str]:
    """Cargo feature names required on the lab host for `flag`."""
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


class QuarkBenchConfigStep(SetupStep):
    """Ensure default (non-TSOT) Quark config for standard benchmarks."""

    name = "quark-bench-config"

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        r = remote.sh(deploy_config_script(bench_config_json()), timeout=60, stream=stream)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        return StepResult(
            self.name,
            True,
            "bench config (UringIO)",
        )


class QuarkExperimentalConfigStep(SetupStep):
    """Deploy Group-5 A/B config with one experimental I/O flag enabled."""

    name = "quark-experimental-config"

    def __init__(self, flag: str) -> None:
        self.flag = flag

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        cfg = experimental_config_json(self.flag)
        features = " ".join(experimental_cargo_features(self.flag))
        r = remote.sh(deploy_config_script(cfg), timeout=60, stream=stream)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        return StepResult(
            self.name,
            True,
            f"experimental config {self.flag}=true (build with: CARGO_FEATURES={features})",
        )
