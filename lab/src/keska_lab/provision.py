"""Python-native Quark lab provisioning — all server setup via SSH."""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from keska_lab.config import LabConfig
from keska_lab.remote import RemoteHost, RemoteResult
from keska_lab.setup.base import SetupReport, StepResult
from keska_lab.textutil import tail_lines

PINNED_TOOLCHAIN = "nightly-2024-07-01-x86_64-unknown-linux-gnu"

APT_PACKAGES = (
    "build-essential",
    "git",
    "curl",
    "pkg-config",
    "libssl-dev",
    "libcap-dev",
    "libudev-dev",
    "libnl-3-dev",
    "libnl-route-3-dev",
    "protobuf-compiler",
    "clang",
    "llvm",
    "binutils",
)


class ProvisionError(RuntimeError):
    pass


@dataclass
class ProvisionContext:
    remote: RemoteHost
    config: LabConfig
    repo: Path
    profile: str = "release"


def _step(name: str, fn: Callable[[], str]) -> StepResult:
    t0 = time.perf_counter()
    try:
        msg = fn()
        return StepResult(name, True, msg, time.perf_counter() - t0)
    except ProvisionError as e:
        return StepResult(name, False, str(e), time.perf_counter() - t0)
    except Exception as e:
        return StepResult(name, False, str(e), time.perf_counter() - t0)


def _fail(result: RemoteResult, remote: RemoteHost, context: str = "") -> None:
    raise ProvisionError(remote.format_failure(result) or context)


def preflight(ctx: ProvisionContext) -> str:
    r = ctx.remote.sh("uname -m && test -c /dev/kvm && echo KVM_OK", timeout=30)
    if not r.ok or "KVM_OK" not in r.stdout:
        raise ProvisionError(ctx.remote.format_failure(r) or "KVM not available on lab")
    return r.stdout.strip().splitlines()[-1]


def ensure_sudo(ctx: ProvisionContext) -> str:
    from keska_lab.sudo import HostAccessError, require_host_access

    try:
        return require_host_access(ctx.remote, ctx.config.sudo_password)
    except HostAccessError as e:
        raise ProvisionError(str(e)) from e


def ensure_apt_deps(ctx: ProvisionContext) -> str:
    pkgs = " ".join(APT_PACKAGES)
    script = f"""
    set -euo pipefail
    need=""
    for pkg in {pkgs}; do
      dpkg -s "$pkg" >/dev/null 2>&1 || need="$need $pkg"
    done
    if [ -z "$need" ]; then
      echo deps_ok
      exit 0
    fi
    if ! sudo -n true 2>/dev/null; then
      echo "missing packages:$need (sudo required)" >&2
      exit 1
    fi
    sudo -n DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $need
    echo deps_ok
    """
    r = ctx.remote.sh(script, timeout=900)
    if not r.ok:
        raise ProvisionError(ctx.remote.format_failure(r))
    return "apt packages OK"


def ensure_rust(ctx: ProvisionContext) -> str:
    tc = PINNED_TOOLCHAIN
    script = f"""
    set -euo pipefail
    export PATH="$HOME/.cargo/bin:$PATH"
    if ! command -v rustup >/dev/null; then
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    fi
    if ! rustup toolchain list | grep -q "^{tc}"; then
      rustup toolchain install {tc}
    fi
    rustup component add rust-src --toolchain {tc}
    rustup default {tc}
    test -f "$HOME/.rustup/toolchains/{tc}/lib/rustlib/src/rust/Cargo.lock"
    rustc +{tc} --version
    """
    r = ctx.remote.sh(script, timeout=900)
    if not r.ok:
        raise ProvisionError(ctx.remote.format_failure(r))
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else "rust OK"


def sync_sources(ctx: ProvisionContext) -> str:
    ctx.remote.rsync_to_lab(ctx.repo)
    return f"rsync → {ctx.config.remote_repo}"


def clean_vdso(ctx: ProvisionContext) -> str:
    repo = shlex.quote(ctx.config.remote_repo)
    r = ctx.remote.sh(
        f"rm -f {repo}/vdso/*.d {repo}/vdso/*.o && make -C {repo}/vdso clean",
        timeout=60,
    )
    if not r.ok:
        raise ProvisionError(ctx.remote.format_failure(r))
    return "vdso artifacts cleaned"


def build_quark(ctx: ProvisionContext, *, cargo_features: str | None = None) -> str:
    repo = shlex.quote(ctx.config.remote_repo)
    target = "debug" if ctx.profile == "debug" else "release"
    tc = PINNED_TOOLCHAIN
    features = (cargo_features if cargo_features is not None else ctx.config.cargo_features).strip()
    qkernel_only = {"experimental-mmap-read", "experimental-uring-statx", "experimental-io"}
    feat_list = [f.strip() for f in features.split(",") if f.strip()] if features else []
    qk_feats = [f for f in feat_list if f in qkernel_only]
    unknown = [f for f in feat_list if f not in qkernel_only]
    if unknown:
        raise ProvisionError(f"unknown Cargo features: {', '.join(unknown)}")

    if not feat_list:
        build_body = f"make {target}"
    elif qk_feats:
        qk = shlex.quote(",".join(qk_feats))
        build_body = f"make qvisor_{target} && make qkernel_{target} CARGO_FEATURES={qk} && make -C vdso"
    else:
        build_body = f"make {target}"

    script = f"""
    set -euo pipefail
    export PATH="$HOME/.cargo/bin:$PATH"
    cd {repo}
    rm -f vdso/*.d vdso/*.o
    make -C vdso clean
    {build_body}
    """
    r = ctx.remote.sh(script, timeout=3600)
    if not r.ok:
        raise ProvisionError(ctx.remote.format_failure(r))
    tag = f" ({features})" if features else ""
    return f"make {target} OK{tag}"


def install_quark(ctx: ProvisionContext) -> str:
    repo = shlex.quote(ctx.config.remote_repo)
    script = f"""
    set -euo pipefail
    cd {repo}
    if [ ! -f vdso/vdso.so ]; then
      make -C vdso
    fi
    sudo -n make install
    sudo -n mkdir -p /var/log/quark
    """
    r = ctx.remote.sh(script, timeout=600)
    if not r.ok:
        raise ProvisionError(ctx.remote.format_failure(r))
    return "installed quark + config"


def configure_docker(ctx: ProvisionContext) -> str:
    from keska_lab.setup.docker import DockerRuntimeStep

    result = DockerRuntimeStep(ctx.config).run(ctx.remote)
    if not result.ok:
        raise ProvisionError(result.message)
    return result.message or "docker runtimes OK"


def pull_bench_image(ctx: ProvisionContext) -> str:
    from keska_lab.setup.docker import DockerPullStep
    from keska_lab.setup.image_registry import ensure_image_registry_auth

    ensure_image_registry_auth(ctx.remote, ctx.config)
    result = DockerPullStep(ctx.config.bench_image, ctx.config).run(ctx.remote)
    if not result.ok:
        raise ProvisionError(result.message)
    return result.message


def quark_runtime_check(ctx: ProvisionContext) -> str:
    from keska_lab.setup.docker import QuarkRuntimeCheckStep

    result = QuarkRuntimeCheckStep().run(ctx.remote)
    if not result.ok:
        raise ProvisionError(result.message)
    return result.message


def cleanup_sandboxes(ctx: ProvisionContext) -> str:
    from keska_lab.setup.quark_cleanup import cleanup_quark_sandboxes

    cleanup_quark_sandboxes(ctx.remote)
    return "stopped stray sandboxes"


def deploy_quark_bench_config(ctx: ProvisionContext) -> str:
    from keska_lab.setup.quark_config import QuarkBenchConfigStep

    result = QuarkBenchConfigStep().run(ctx.remote)
    if not result.ok:
        raise ProvisionError(result.message)
    return result.message or "quark bench config deployed"


def ensure_oci_bundle_step(ctx: ProvisionContext) -> str:
    from keska_lab.setup.oci_bundle import ensure_oci_bundle

    path = ensure_oci_bundle(ctx.remote, ctx.config, ctx.config.bench_image)
    return f"OCI bundle at {path}"


def smoke_test(ctx: ProvisionContext) -> str:
    from keska_lab.backends.quark import QuarkBackend
    from keska_lab.setup.oci_bundle import bundle_dir, verify_bundle_markers

    path = bundle_dir(ctx.config, ctx.config.bench_image)
    verify_bundle_markers(ctx.remote, path, ctx.config.bench_image)

    backend = QuarkBackend(
        ctx.remote,
        profile=ctx.config.quark_build_profile,
        exec_mode="direct",
    )
    ms = backend.tti_once(image=ctx.config.bench_image)
    backend.cleanup()
    return f"direct OCI smoke OK ({ms:.0f} ms)"


def provision_quark(
    remote: RemoteHost,
    config: LabConfig | None = None,
    *,
    skip_build: bool | None = None,
) -> SetupReport:
    """Full pristine Quark build + install on lab."""
    import os

    cfg = config or remote.config
    if skip_build is None:
        skip_build = os.environ.get("KESKA_LAB_SKIP_BUILD", "").lower() in ("1", "true", "yes")
    ctx = ProvisionContext(
        remote=remote,
        config=cfg,
        repo=cfg.resolve_local_repo(),
        profile=cfg.quark_build_profile,
    )

    steps: list[tuple[str, Callable[[], str]]] = [
        ("preflight", lambda: preflight(ctx)),
        ("bootstrap-host", lambda: ensure_sudo(ctx)),
        ("cleanup-sandboxes", lambda: cleanup_sandboxes(ctx)),
        ("ensure-apt-deps", lambda: ensure_apt_deps(ctx)),
        ("ensure-rust", lambda: ensure_rust(ctx)),
    ]
    if not skip_build:
        steps.extend(
            [
                ("sync-sources", lambda: sync_sources(ctx)),
                ("clean-vdso", lambda: clean_vdso(ctx)),
                ("build-quark", lambda: build_quark(ctx)),
            ]
        )
    steps.extend(
        [
            ("install-quark", lambda: install_quark(ctx)),
            ("quark-bench-config", lambda: deploy_quark_bench_config(ctx)),
            ("oci-bundle", lambda: ensure_oci_bundle_step(ctx)),
            ("smoke-test", lambda: smoke_test(ctx)),
        ]
    )
    if cfg.quark_exec_mode == "docker":
        steps.extend(
            [
                ("docker-runtimes", lambda: configure_docker(ctx)),
                ("quark-runtime-check", lambda: quark_runtime_check(ctx)),
            ]
        )

    report = SetupReport(pipeline="quark-provision")
    for name, fn in steps:
        result = _step(name, fn)
        report.steps.append(result)
        if not result.ok:
            break
    return report


def main() -> int:
    from keska_lab.display import print_setup_report

    remote = RemoteHost()
    report = provision_quark(remote)
    print_setup_report(report)
    if not report.ok:
        failed = next(s for s in report.steps if not s.ok)
        print(f"\nFailed at {failed.name}:\n{tail_lines(failed.message, 30)}", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
