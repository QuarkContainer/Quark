"""Lab host privilege bootstrap — passwordless sudo and docker group."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from keska_lab.remote import RemoteHost


class HostAccessError(RuntimeError):
    pass


@dataclass
class HostAccess:
    passwordless_sudo: bool
    docker_group: bool


def probe_host_access(remote: RemoteHost) -> HostAccess:
    sudo_ok = remote.sh("sudo -n true", timeout=15).ok
    groups = remote.sh("id -nG", timeout=15).stdout.split()
    return HostAccess(passwordless_sudo=sudo_ok, docker_group="docker" in groups)


def bootstrap_host_access(remote: RemoteHost, sudo_password: str) -> str:
    """One-time setup: NOPASSWD sudo + docker group membership for lab user."""
    pw = shlex.quote(sudo_password)
    script = f"""
    set -euo pipefail
    SUDO() {{ echo {pw} | sudo -S "$@"; }}
    SUDO -v
    echo 'lab ALL=(ALL) NOPASSWD: ALL' | SUDO tee /etc/sudoers.d/keska-lab >/dev/null
    SUDO chmod 440 /etc/sudoers.d/keska-lab
    SUDO usermod -aG docker lab
    """
    r = remote.sh(script, timeout=120)
    if not r.ok:
        raise HostAccessError(remote.format_failure(r) or "host bootstrap failed")
    verify = remote.sh("sudo -n true", timeout=15)
    if not verify.ok:
        raise HostAccessError(
            "passwordless sudo not active after bootstrap.\n"
            "Check /etc/sudoers.d/keska-lab on lab:\n"
            "  sudo visudo -cf /etc/sudoers.d/keska-lab"
        )
    access = probe_host_access(remote)
    if not access.docker_group:
        # group added but current session may not reflect it; verify via getent
        r2 = remote.sh("getent group docker | grep -q '\\blab\\b'", timeout=15)
        if not r2.ok:
            raise HostAccessError("lab user not in docker group after bootstrap")
    return "NOPASSWD sudo + docker group configured"


def require_host_access(remote: RemoteHost, sudo_password: str | None) -> str:
    access = probe_host_access(remote)
    if access.passwordless_sudo and access.docker_group:
        return "host access OK"
    if access.passwordless_sudo and not access.docker_group:
        r = remote.sh("sudo -n usermod -aG docker lab", timeout=30)
        if r.ok:
            return "added lab to docker group"
        raise HostAccessError(
            "passwordless sudo OK but lab not in docker group.\n"
            "Run on lab: sudo usermod -aG docker lab"
        )
    if sudo_password:
        return bootstrap_host_access(remote, sudo_password)
    raise HostAccessError(
        "lab host needs passwordless sudo and docker group access.\n"
        "One-time fix (on lab, as a user with sudo):\n"
        "  echo 'lab ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/keska-lab\n"
        "  sudo chmod 440 /etc/sudoers.d/keska-lab\n"
        "  sudo usermod -aG docker lab\n"
        "Or re-run with: KESKA_LAB_SUDO_PASSWORD='...' keska-lab-provision"
    )


def main() -> int:
    import sys

    from keska_lab.config import LabConfig
    from keska_lab.remote import RemoteHost

    password = LabConfig.from_env().sudo_password
    if not password:
        print(
            "Set KESKA_LAB_SUDO_PASSWORD to the lab user's sudo password.",
            file=sys.stderr,
        )
        return 2
    remote = RemoteHost()
    try:
        msg = bootstrap_host_access(remote, password)
        print(msg)
        return 0
    except HostAccessError as e:
        print(str(e), file=sys.stderr)
        return 1
