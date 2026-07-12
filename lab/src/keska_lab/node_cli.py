"""CLI for lab node install / verify / gate / cleanup."""

from __future__ import annotations

import argparse
import sys

from keska_lab.config import LabConfig
from keska_lab.display import console
from keska_lab.gate.runner import NodeGateRunner
from keska_lab.installer import InstallOptions, NodeInstaller
from keska_lab.knode import KNode, install_node
from keska_lab.profile import NetworkMode, NodeProfile


def _profile_from_args(args) -> NodeProfile:
    if args.profile:
        mapping = {
            "quark_bridge": NodeProfile.quark_bridge,
            "quark_tsot": NodeProfile.quark_tsot,
            "kata_bridge": NodeProfile.kata_bridge,
        }
        factory = mapping.get(args.profile)
        if factory is None:
            raise SystemExit(f"unknown profile {args.profile!r}")
        return factory()
    cfg = LabConfig.from_env()
    if args.network:
        cfg.network_mode = NetworkMode(args.network)
    if args.runtime:
        prof = NodeProfile(
            runtime=args.runtime,
            network=cfg.network_mode,
            build_profile=cfg.quark_build_profile,
            bench_image=cfg.bench_image,
        )
        prof.validate()
        return prof
    return cfg.node_profile


def cmd_install(args) -> int:
    cfg = LabConfig.from_env()
    if getattr(args, "skip_registry_auth", False):
        cfg.skip_registry_auth = True
    profile = _profile_from_args(args)
    opts = InstallOptions(
        gate_level=args.gate,
        network_only=getattr(args, "network_only", False),
        skip_provision=getattr(args, "skip_provision", False),
        skip_containerd=getattr(args, "skip_containerd", False),
        skip_gates=getattr(args, "skip_gates", False),
    )
    node = install_node(cfg, profile, options=opts)
    st = node.status()
    console.print(f"[green]installed[/green] {profile.name} vms={st.vm_count} health_ok={st.health.ok}")
    return 0


def cmd_verify(args) -> int:
    cfg = LabConfig.from_env()
    profile = _profile_from_args(args)
    from keska_lab.remote import RemoteHost

    report = NodeInstaller(RemoteHost(cfg), profile).verify(
        network_only=getattr(args, "network_only", False)
    )
    for c in report.checks:
        color = "green" if c.status.value in ("ok", "skip") else "yellow" if c.status.value == "degraded" else "red"
        console.print(f"[{color}]{c.name}[/{color}] {c.status.value}: {c.message}")
    return 0 if report.ok else 1


def cmd_gate(args) -> int:
    cfg = LabConfig.from_env()
    profile = _profile_from_args(args)
    from keska_lab.remote import RemoteHost

    results = NodeGateRunner(RemoteHost(cfg), profile).run(args.gate)
    ok = True
    for r in results:
        color = "green" if r.ok else "red"
        console.print(f"[{color}]{r.gate}[/{color}] {r.message} ({r.duration_s:.1f}s)")
        ok = ok and r.ok
    return 0 if ok else 1


def cmd_cleanup(args) -> int:
    cfg = LabConfig.from_env()
    profile = _profile_from_args(args)
    from keska_lab.remote import RemoteHost

    NodeInstaller(RemoteHost(cfg), profile).cleanup()
    console.print("[green]cleanup complete[/green]")
    return 0


def cmd_status(args) -> int:
    cfg = LabConfig.from_env()
    profile = _profile_from_args(args)
    from keska_lab.remote import RemoteHost

    node = KNode(RemoteHost(cfg), profile)
    st = node.status()
    console.print(f"profile={st.profile.name} vms={st.vm_count} health_ok={st.health.ok}")
    for c in st.health.checks:
        console.print(f"  {c.name}: {c.status.value} — {c.message}")
    return 0 if st.ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Keska lab node install / verify")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument(
            "--profile",
            choices=("quark_bridge", "quark_tsot", "kata_bridge"),
            help="preset NodeProfile",
        )
        sp.add_argument("--runtime", choices=("quark", "kata"))
        sp.add_argument("--network", choices=("bridge", "tsot"))

    sp = sub.add_parser("install", help="Install host for profile")
    add_common(sp)
    sp.add_argument("--gate", default="L1", help="Gate level (L1)")
    sp.add_argument(
        "--skip-registry-auth",
        action="store_true",
        help="Skip GCP artifact registry login (non-interactive CI)",
    )
    sp.add_argument(
        "--network-only",
        action="store_true",
        help="Reconfigure CNI/TSOT/quark config only (no provision/containerd)",
    )
    sp.add_argument("--skip-provision", action="store_true")
    sp.add_argument("--skip-containerd", action="store_true")
    sp.add_argument("--skip-gates", action="store_true")
    sp.set_defaults(func=cmd_install)

    sp = sub.add_parser("verify", help="Verify host matches profile")
    add_common(sp)
    sp.add_argument(
        "--network-only",
        action="store_true",
        help="Check CNI/TSOT/quark config only (skip containerd/crictl)",
    )
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("gate", help="Run functional gates")
    add_common(sp)
    sp.add_argument("--gate", default="L1")
    sp.set_defaults(func=cmd_gate)

    sp = sub.add_parser("cleanup", help="Cleanup profile services")
    add_common(sp)
    sp.set_defaults(func=cmd_cleanup)

    sp = sub.add_parser("status", help="Show node status")
    add_common(sp)
    sp.set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        console.print(f"[red]error[/red]: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
