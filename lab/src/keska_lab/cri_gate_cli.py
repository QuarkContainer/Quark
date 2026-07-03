"""CLI: keska-lab-cri-gate — CRI validation layers for Quark shim."""

from __future__ import annotations

import argparse
import sys

from keska_lab.config import LabConfig
from keska_lab.cri.gate import run_layer, run_layers_through
from keska_lab.cri.kata_setup import ensure_kata_cri_ready
from keska_lab.remote import RemoteHost
from keska_lab.setup.containerd_cri import ContainerdCriStep, CrictlInstallStep


def _needs_kata_setup(runtime: str, parity: bool) -> bool:
    return parity or runtime == "kata"


def _run_kata_setup(cfg: LabConfig, remote: RemoteHost) -> int:
    print(f"Kata CRI setup on {cfg.ssh_target}…")
    for res in ensure_kata_cri_ready(remote, cfg):
        print(f"  {res.name}: {'ok' if res.ok else 'FAIL'} — {res.message}")
        if not res.ok:
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Quark CRI validation gate (L1–L5)")
    p.add_argument(
        "--layer",
        default="L2",
        help="Highest layer to run through (L1, L2, L3, L5). Default L2.",
    )
    p.add_argument(
        "--runtime",
        default="quark",
        help="crictl runtime handler: quark (default), kata, or empty for default",
    )
    p.add_argument(
        "--setup",
        action="store_true",
        help="Run containerd CRI + crictl install before gate",
    )
    p.add_argument(
        "--no-config",
        action="store_true",
        help="Skip deploying /etc/quark/config.json (cri profile)",
    )
    p.add_argument(
        "--parity",
        action="store_true",
        help="Run --layer through quark then kata (L6-style)",
    )
    args = p.parse_args(argv)

    cfg = LabConfig.from_env()
    remote = RemoteHost(cfg)

    if args.setup:
        print(f"CRI setup on {cfg.ssh_target}…")
        for step in (ContainerdCriStep(), CrictlInstallStep()):
            res = step.run(remote)
            print(f"  {step.name}: {'ok' if res.ok else 'FAIL'} — {res.message}")
            if not res.ok:
                return 1
        if _needs_kata_setup(args.runtime, args.parity):
            if _run_kata_setup(cfg, remote):
                return 1

    elif _needs_kata_setup(args.runtime, args.parity):
        if _run_kata_setup(cfg, remote):
            return 1

    deploy = not args.no_config

    if args.parity:
        ok = True
        for rt in ("quark", "kata"):
            print(f"\n=== parity runtime={rt} layer<={args.layer} ===")
            results = run_layers_through(
                remote,
                args.layer,
                runtime=rt,
                deploy_quark_cri_config=deploy and rt == "quark",
            )
            for r in results:
                print(f"  {r.layer}: {'PASS' if r.ok else 'FAIL'} — {r.message}")
                if not r.ok:
                    ok = False
                    break
        return 0 if ok else 1

    results = run_layers_through(
        remote,
        args.layer,
        runtime=args.runtime,
        deploy_quark_cri_config=deploy,
    )
    for r in results:
        print(f"{r.layer}: {'PASS' if r.ok else 'FAIL'} — {r.message}")
    return 0 if results and results[-1].ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
