"""Kata Containers install on Ubuntu (no distro packages on 24.04)."""

from __future__ import annotations

import textwrap

from keska_lab.remote import RemoteHost

KATA_VERSION = "3.15.0"
KATA_ROOT = "/opt/kata"


def kata_install_script(version: str = KATA_VERSION) -> str:
    pkg = f"kata-static-{version}-amd64.tar.xz"
    url = f"https://github.com/kata-containers/kata-containers/releases/download/{version}/{pkg}"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        if [ -x {KATA_ROOT}/bin/kata-runtime ]; then
          echo {KATA_ROOT}/bin/kata-runtime
          exit 0
        fi
        if [ ! -f /var/tmp/{pkg} ]; then
          wget -q -O /var/tmp/{pkg} {url!r}
        fi
        sudo -n rm -rf {KATA_ROOT}
        sudo -n mkdir -p {KATA_ROOT}
        sudo -n tar -xJf /var/tmp/{pkg} -C /opt
        if [ -d /opt/opt/kata ]; then
          sudo -n mv /opt/opt/kata {KATA_ROOT}
          sudo -n rmdir /opt/opt 2>/dev/null || sudo -n rm -rf /opt/opt
        fi
        test -x {KATA_ROOT}/bin/kata-runtime
        sudo -n ln -sf {KATA_ROOT}/share/defaults/kata-containers /etc/kata-containers
        sudo -n ln -sf {KATA_ROOT}/bin/kata-runtime /usr/local/bin/kata-runtime
        sudo -n ln -sf {KATA_ROOT}/bin/containerd-shim-kata-v2 /usr/local/bin/containerd-shim-kata-v2
        sudo -n ln -sf {KATA_ROOT}/bin/kata-runtime /usr/bin/kata-runtime
        {KATA_ROOT}/bin/kata-runtime --version | head -1
        """
    ).strip()


def ensure_kata(remote: RemoteHost, *, stream: bool = False) -> str:
    r = remote.sh(kata_install_script(), timeout=900, stream=stream)
    if not r.ok or "kata-runtime" not in (r.stdout + r.stderr).lower():
        raise RuntimeError(remote.format_failure(r) or "Kata install failed")
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else "kata installed"
