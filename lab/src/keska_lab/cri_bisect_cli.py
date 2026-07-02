"""CLI: keska-lab-cri-bisect — per-bug negative proof on lab."""

from __future__ import annotations

import argparse
import subprocess
import sys

from keska_lab.config import LabConfig
from keska_lab.cri.bisect import (
    BUG_CHAIN,
    backup_fixed_tree,
    check_032_local,
    restore_fixed_files,
    revert_files_to_baseline,
    touch_rebuild_hints,
)
from keska_lab.cri.gate import run_layer
from keska_lab.remote import RemoteHost
from keska_lab.setup.quark_config import cri_bench_config_json, deploy_config_script


def _provision(*, qkernel: bool) -> int:
    del qkernel  # full provision rebuilds qkernel when qkernel/ changed
    print("  provision: python -m keska_lab.provision")
    return subprocess.call([sys.executable, "-m", "keska_lab.provision"])


def _deploy_cri_config(remote: RemoteHost) -> None:
    r = remote.sh(deploy_config_script(cri_bench_config_json()), timeout=60, stream=False)
    if not r.ok:
        raise RuntimeError(f"cri config deploy failed: {r.stderr or r.stdout}")


def _prove_one(bug_id: str, *, skip_provision: bool) -> int:
    entry = next((b for b in BUG_CHAIN if b.bug_id == bug_id), None)
    if entry is None:
        print(f"unknown bug {bug_id}; choose from {[b.bug_id for b in BUG_CHAIN]}")
        return 1

    if entry.bug_id == "032":
        ok, msg = check_032_local()
        print(f"032 local: {'PASS' if ok else 'FAIL'} — {msg}")
        return 0 if ok else 1

    backup_fixed_tree()
    cfg = LabConfig.from_env()
    remote = RemoteHost(cfg)

    print(f"\n=== prove bug {entry.bug_id} — revert {len(entry.files)} file(s) ===")
    try:
        revert_files_to_baseline(entry.files)
        if not skip_provision:
            if _provision(qkernel=entry.rebuild_qkernel):
                return 1

        _deploy_cri_config(remote)
        fail = run_layer(
            remote,
            entry.layer,
            runtime="quark",
            deploy_quark_cri_config=False,
            stream=True,
        )
        print(
            f"  without fix: {fail.layer} {'FAIL' if not fail.ok else 'UNEXPECTED PASS'} — {fail.message}"
        )
        if fail.ok:
            print(f"ERROR: reverting {entry.bug_id} should fail {entry.layer}")
            return 1

        restore_fixed_files(entry.files)
        touch_rebuild_hints(entry.files)
        if not skip_provision:
            if _provision(qkernel=entry.rebuild_qkernel):
                return 1

        _deploy_cri_config(remote)
        ok = run_layer(
            remote,
            entry.layer,
            runtime="quark",
            deploy_quark_cri_config=False,
            stream=True,
        )
        print(f"  with fix restored: {ok.layer} {'PASS' if ok.ok else 'FAIL'} — {ok.message}")
        if not ok.ok:
            return 1

        print(f"PROVEN {entry.bug_id}: without fix fails ({entry.symptom}); with fix passes {entry.layer}")
        return 0
    finally:
        restore_fixed_files(entry.files)
        touch_rebuild_hints(entry.files)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-bug CRI fix proof (negative revert test on lab)")
    p.add_argument("--bug", help="Prove one bug (032–043)")
    p.add_argument("--all", action="store_true", help="Prove entire chain (slow)")
    p.add_argument(
        "--skip-provision",
        action="store_true",
        help="Skip rebuild/deploy between revert/restore (use when binaries already match tree)",
    )
    p.add_argument("--list", action="store_true", help="List bugs and gate layers")
    p.add_argument("--backup", action="store_true", help="Snapshot fixed files to .keska-cri-fix-backup/")
    args = p.parse_args(argv)

    if args.list:
        for b in BUG_CHAIN:
            rk = " +qkernel" if b.rebuild_qkernel else ""
            print(f"{b.bug_id}  layer={b.layer}{rk}  {b.symptom}")
            for f in b.files:
                print(f"    {f}")
        return 0

    if args.backup:
        backup_fixed_tree()
        print("Backed up fixed CRI files to .keska-cri-fix-backup/")
        return 0

    if args.all:
        backup_fixed_tree()
        rc = 0
        for b in BUG_CHAIN:
            if _prove_one(b.bug_id, skip_provision=args.skip_provision):
                rc = 1
        return rc

    if args.bug:
        backup_fixed_tree()
        return _prove_one(args.bug, skip_provision=args.skip_provision)

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
