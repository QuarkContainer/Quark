from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path


def _is_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def ensure_ss_running(
    *,
    repo: Path | None = None,
    port: int = 8890,
    config_path: str = "/etc/quark/lab-qlet.json",
    timeout_s: float = 10.0,
) -> None:
    """Ensure `ss` is listening on the expected port.

    This is a *temporary* stabilizer until TSOT install/health are made strict again.
    It is bounded and only starts ss when it appears missing.
    """
    if _is_listening(port):
        return

    repo = repo or (Path.home() / "Quark")
    candidates = [
        repo / "qservice" / "target" / "release" / "ss",
        repo / "qservice" / "target" / "debug" / "ss",
    ]
    ss_bin = next((p for p in candidates if p.exists()), None)
    if ss_bin is None:
        raise RuntimeError(f"ss binary not found under {repo}/qservice/target")

    # Stop any old ss and start a new one.
    subprocess.run(
        ["sudo", "-n", "pkill", "-x", "ss"],
        text=True,
        capture_output=True,
        timeout=2,
        check=False,
    )
    subprocess.run(
        ["sudo", "-n", "pkill", "-f", "qservice/target/.*/ss"],
        text=True,
        capture_output=True,
        timeout=2,
        check=False,
    )

    subprocess.run(
        ["sudo", "-n", "nohup", str(ss_bin), config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
        timeout=2,
    )

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _is_listening(port):
            return
        time.sleep(0.5)
    raise TimeoutError(f"ss not listening on {port} after {timeout_s}s")

