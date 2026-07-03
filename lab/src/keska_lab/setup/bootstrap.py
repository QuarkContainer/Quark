"""Lab host build prerequisites — apt packages + Rust."""

from __future__ import annotations

import textwrap

from keska_lab.provision import APT_PACKAGES, PINNED_TOOLCHAIN
from keska_lab.setup.base import SetupStep, StepResult
from keska_lab.remote import RemoteHost


def _install_script() -> str:
    pkgs = " ".join(APT_PACKAGES)
    tc = PINNED_TOOLCHAIN
    return textwrap.dedent(
        f"""
        set -euo pipefail
        export PATH="$HOME/.cargo/bin:$PATH"

        need=""
        for pkg in {pkgs}; do
          dpkg -s "$pkg" >/dev/null 2>&1 || need="$need $pkg"
        done
        if [ -n "$need" ]; then
          sudo -n DEBIAN_FRONTEND=noninteractive apt-get update -qq
          sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $need
        fi

        if ! command -v rustup >/dev/null; then
          curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
          export PATH="$HOME/.cargo/bin:$PATH"
        fi

        if ! rustup toolchain list | grep -q "^{tc}"; then
          rustup toolchain install {tc}
        fi
        rustup component add rust-src --toolchain {tc}
        rustup default {tc}

        test -f "$HOME/.rustup/toolchains/{tc}/lib/rustlib/src/rust/Cargo.lock"
        pkg-config --exists libcap
        rustc +{tc} --version
        """
    ).strip()


class EnsureQuarkBuildEnvStep(SetupStep):
    """Install apt build deps + pinned Rust toolchain on the lab host."""

    name = "ensure-build-env"

    def run(self, remote: RemoteHost) -> StepResult:
        r = remote.sh(_install_script(), timeout=900, check=False)
        combined = (r.stdout or "") + (r.stderr or "")
        ok = r.ok and "rustc" in combined.lower()
        if ok:
            last = [ln for ln in combined.splitlines() if ln.strip()]
            return StepResult(self.name, True, last[-1] if last else "build env ready")
        hint = remote.format_failure(r)
        if "sudo" in combined.lower() or "password" in combined.lower():
            hint += (
                "\n\nPasswordless sudo required on lab, or run manually:\n"
                f"  sudo apt-get install -y {' '.join(APT_PACKAGES)}"
            )
        elif "rust-src" in combined.lower() or "cargo.lock" in combined.lower():
            hint += (
                f"\n\nQuark needs rust-src on the pinned toolchain:\n"
                f"  rustup toolchain install {PINNED_TOOLCHAIN}\n"
                f"  rustup component add rust-src --toolchain {PINNED_TOOLCHAIN}"
            )
        elif "libcap" in combined.lower() or "-lcap" in combined.lower():
            hint += "\n\nMissing libcap-dev. Run on lab: sudo apt-get install -y libcap-dev"
        return StepResult(self.name, False, hint)


class BootstrapLabStep(EnsureQuarkBuildEnvStep):
    """Alias for %bootstrap magic."""

    name = "bootstrap-lab"
