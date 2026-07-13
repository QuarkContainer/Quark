from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


class CriError(RuntimeError):
    pass


@dataclass(frozen=True)
class CriResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CriClient:
    """Minimal CRI client using crictl (fail-fast, no shell=True)."""

    def __init__(self, *, sudo: bool = True):
        self.sudo = sudo

    def _run(self, argv: list[str], *, timeout_s: float) -> CriResult:
        cmd = (["sudo", "-n"] if self.sudo else []) + argv
        try:
            p = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"crictl timeout after {timeout_s}s: {cmd}") from e
        return CriResult(p.stdout, p.stderr, p.returncode)

    def runp(self, pod_json_path: str, *, runtime_handler: str = "", timeout_s: float = 45.0) -> str:
        argv = ["crictl", "runp"]
        if runtime_handler:
            argv.append(f"--runtime={runtime_handler}")
        argv.append(pod_json_path)
        r = self._run(argv, timeout_s=timeout_s)
        if not r.ok:
            raise CriError(f"crictl runp failed: rc={r.returncode} err={r.stderr.strip()}")
        return r.stdout.strip()

    def run(self, container_json_path: str, pod_json_path: str, *, timeout_s: float = 60.0) -> str:
        r = self._run(["crictl", "run", container_json_path, pod_json_path], timeout_s=timeout_s)
        if not r.ok:
            raise CriError(f"crictl run failed: rc={r.returncode} err={r.stderr.strip()}")
        return r.stdout.strip()

    def create(
        self,
        pod_id: str,
        container_json_path: str,
        pod_json_path: str,
        *,
        timeout_s: float = 30.0,
    ) -> str:
        argv = ["crictl", "create", "--no-pull", pod_id, container_json_path, pod_json_path]
        r = self._run(argv, timeout_s=timeout_s)
        if not r.ok:
            raise CriError(f"crictl create failed: rc={r.returncode} err={r.stderr.strip()}")
        return r.stdout.strip()

    def start(self, container_id: str, *, timeout_s: float = 30.0) -> None:
        if not container_id:
            return
        r = self._run(["crictl", "start", container_id], timeout_s=timeout_s)
        if not r.ok:
            raise CriError(f"crictl start failed: rc={r.returncode} err={r.stderr.strip()}")

    def stopp(self, pod_id: str, *, timeout_s: float = 45.0) -> None:
        if not pod_id:
            return
        _ = self._run(["crictl", "stopp", pod_id], timeout_s=timeout_s)

    def rmp(self, pod_id: str, *, force: bool = True, timeout_s: float = 45.0) -> None:
        if not pod_id:
            return
        argv = ["crictl", "rmp"]
        if force:
            argv.append("-f")
        argv.append(pod_id)
        _ = self._run(argv, timeout_s=timeout_s)

    def rm(self, container_id: str, *, force: bool = True, timeout_s: float = 30.0) -> None:
        if not container_id:
            return
        argv = ["crictl", "rm"]
        if force:
            argv.append("-f")
        argv.append(container_id)
        _ = self._run(argv, timeout_s=timeout_s)

    def logs(self, container_id: str, *, tail: int = 200, timeout_s: float = 10.0) -> str:
        argv = ["crictl", "logs", f"--tail={tail}", container_id]
        r = self._run(argv, timeout_s=timeout_s)
        if not r.ok:
            return ""
        return r.stdout

    def inspect(self, container_id: str, *, timeout_s: float = 10.0) -> dict:
        r = self._run(["crictl", "inspect", container_id], timeout_s=timeout_s)
        if not r.ok:
            raise CriError(f"crictl inspect failed: rc={r.returncode} err={r.stderr.strip()}")
        return json.loads(r.stdout)

    def inspectp(self, pod_id: str, *, timeout_s: float = 10.0) -> dict:
        r = self._run(["crictl", "inspectp", pod_id], timeout_s=timeout_s)
        if not r.ok:
            raise CriError(f"crictl inspectp failed: rc={r.returncode} err={r.stderr.strip()}")
        return json.loads(r.stdout)

