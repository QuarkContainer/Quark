"""SSH, rsync, and scp to the lab host."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from keska_lab.config import LabConfig
from keska_lab.textutil import format_cmd_output


@dataclass
class RemoteResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_if_failed(self, context: str = "") -> RemoteResult:
        if not self.ok:
            msg = f"{context}: exit {self.returncode}\n{self.stderr.strip()}"
            raise RuntimeError(msg)
        return self


class RemoteHost:
    """Run commands and transfer files to the lab host."""

    RSYNC_EXCLUDES = (
        "target",
        "build",
        ".git",
        "lab/.venv",
        ".cursor",
        "**/__pycache__",
        "vdso/*.d",
        "vdso/*.o",
        "vdso/*.so",
    )

    def __init__(self, config: LabConfig | None = None):
        self.config = config or LabConfig.from_env()

    @property
    def ssh_target(self) -> str:
        return self.config.ssh_target

    def _ssh_opts(self, *, batch: bool = True) -> list[str]:
        opts = ["-o", "ConnectTimeout=15"]
        if batch:
            opts.append("-o")
            opts.append("BatchMode=yes")
        if self.config.ssh_key:
            opts.extend(["-i", str(self.config.ssh_key)])
        return opts

    def _base_ssh(self, *, batch: bool = True) -> list[str]:
        return ["ssh", *self._ssh_opts(batch=batch), self.ssh_target]

    def _base_rsync(self) -> list[str]:
        cmd = ["rsync", "-az", "--delete"]
        for ex in self.RSYNC_EXCLUDES:
            cmd.extend(["--exclude", ex])
        cmd.extend(["-e", "ssh " + " ".join(self._ssh_opts())])
        return cmd

    def run(
        self,
        remote_cmd: str,
        *,
        timeout: int | None = 600,
        check: bool = False,
        stream: bool = False,
        tty: bool = False,
    ) -> RemoteResult:
        cmd = [*self._base_ssh(batch=not tty), *(["-t"] if tty else []), remote_cmd]
        if stream or tty:
            result = self._run_streaming(cmd, timeout=timeout)
        else:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result = RemoteResult(proc.stdout, proc.stderr, proc.returncode)
        if check:
            result.raise_if_failed(remote_cmd[:120])
        return result

    def run_tty(
        self,
        remote_cmd: str,
        *,
        timeout: int | None = None,
        stream: bool = True,
    ) -> RemoteResult:
        """Interactive SSH session (no BatchMode) for gcloud login etc."""
        return self.run(remote_cmd, timeout=timeout, stream=stream, tty=True)

    def _run_streaming(self, cmd: list[str], *, timeout: int | None) -> RemoteResult:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                lines.append(line)
                sys.stdout.write(line)
                if not line.endswith("\n"):
                    sys.stdout.write("\n")
                sys.stdout.flush()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        out = "".join(lines)
        return RemoteResult(out, "", proc.returncode or 0)

    def _run_script(
        self,
        script: str,
        *,
        timeout: int | None = 600,
        check: bool = False,
        stream: bool = False,
    ) -> RemoteResult:
        """Run a multiline script via SSH stdin (avoids ARG_MAX and ctr stdin steal)."""
        remote_cmd = (
            "tmp=/tmp/keska-sh-$RANDOM.sh && "
            "cat > \"$tmp\" && "
            "bash \"$tmp\"; ec=$?; rm -f \"$tmp\"; exit $ec"
        )
        cmd = [*self._base_ssh(), remote_cmd]
        if stream:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdin is not None
            assert proc.stdout is not None
            proc.stdin.write(script)
            proc.stdin.close()
            lines: list[str] = []
            try:
                for line in proc.stdout:
                    lines.append(line)
                    sys.stdout.write(line)
                    if not line.endswith("\n"):
                        sys.stdout.write("\n")
                    sys.stdout.flush()
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise
            result = RemoteResult("".join(lines), "", proc.returncode or 0)
        else:
            proc = subprocess.run(
                cmd,
                input=script,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result = RemoteResult(proc.stdout, proc.stderr, proc.returncode)
        if check:
            result.raise_if_failed("remote script")
        return result

    def sh(
        self,
        script: str,
        *,
        timeout: int | None = 600,
        check: bool = False,
        stream: bool = False,
    ) -> RemoteResult:
        if "\n" in script:
            return self._run_script(script, timeout=timeout, check=check, stream=stream)
        remote_cmd = f"bash -c {shlex.quote(script)}"
        return self.run(remote_cmd, timeout=timeout, check=check, stream=stream)

    def sh_login(
        self,
        script: str,
        *,
        timeout: int | None = 600,
        check: bool = False,
        stream: bool = False,
    ) -> RemoteResult:
        """Profile PATH without bash -lc (multiline -lc breaks SSH exit codes)."""
        wrapped = (
            "source ~/.profile 2>/dev/null || "
            "source ~/.bash_profile 2>/dev/null || "
            "source ~/.bashrc 2>/dev/null || true\n"
            f"{script}"
        )
        return self.sh(wrapped, timeout=timeout, check=check, stream=stream)

    def format_failure(self, result: RemoteResult) -> str:
        return format_cmd_output(result.stdout, result.stderr)

    def ping(self) -> RemoteResult:
        return self.sh("uname -a && echo OK", timeout=30)

    def which(self, binary: str) -> str | None:
        r = self.sh_login(f"command -v {shlex.quote(binary)} 2>/dev/null || true", timeout=15)
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        return lines[-1] if lines else None

    def docker_sh(
        self,
        script: str,
        *,
        timeout: int | None = 600,
        check: bool = False,
        stream: bool = False,
    ) -> RemoteResult:
        """Run docker commands via sg docker (works before group refresh in SSH session)."""
        quoted = shlex.quote(script)
        return self.sh(f"sg docker -c {quoted}", timeout=timeout, check=check, stream=stream)

    def in_docker_group(self) -> bool:
        r = self.sh("id -nG", timeout=15)
        return "docker" in r.stdout.split()

    def rsync_to_lab(
        self,
        local_path: Path,
        remote_subpath: str = "",
        *,
        stream: bool = False,
    ) -> None:
        """Sync local directory to lab repo path."""
        dest = self.config.remote_repo
        if remote_subpath:
            dest = f"{dest.rstrip('/')}/{remote_subpath}"
        dest = f"{self.ssh_target}:{dest}/"
        local = str(local_path.resolve()) + "/"
        cmd = [*self._base_rsync(), local, dest]
        if stream:
            subprocess.run(cmd, check=True, timeout=3600)
        else:
            subprocess.run(cmd, check=True, timeout=3600, capture_output=True, text=True)

    def scp_to_lab(self, local_file: Path, remote_path: str) -> None:
        """Copy a single file to the lab host."""
        dest = f"{self.ssh_target}:{remote_path}"
        subprocess.run(
            ["scp", *self._ssh_opts(), str(local_file), dest],
            check=True,
            timeout=600,
        )

    def scp_tree_to_lab(self, local_dir: Path, remote_dir: str) -> None:
        """Rsync a local directory to an arbitrary remote path."""
        dest = f"{self.ssh_target}:{remote_dir.rstrip('/')}/"
        subprocess.run(
            [*self._base_rsync(), str(local_dir.resolve()) + "/", dest],
            check=True,
            timeout=3600,
        )

    def ensure_workdir(self) -> None:
        self.sh(f"mkdir -p {shlex.quote(self.config.work_dir)}", check=True)
