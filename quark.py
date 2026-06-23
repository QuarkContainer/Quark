#!/usr/bin/env python3
"""
Quark development CLI — single entry point for macOS + Lima dev workflow.

  python3 quark.py help
  python3 quark.py doctor
  python3 quark.py dev start
  python3 quark.py rebuild
  python3 quark.py bench run --profile dev

Shell scripts in scripts/dev/bin/ implement VM operations (limactl, docker, rsync).
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEV_BIN = REPO / "scripts" / "dev" / "bin"
BENCH = REPO / "benchmark"
LIMA = os.environ.get("LIMA_INSTANCE", "quark")
LIMA_YAML = REPO / "lima-quark.yaml"
RUNTIME = os.environ.get("RUNTIME", "quark_d")
DOCKER_FLAGS = os.environ.get("DOCKER_FLAGS", "--rm")


def env(**overrides: str) -> dict[str, str]:
    e = os.environ.copy()
    e.update({k: str(v) for k, v in overrides.items()})
    return e


def confirmed(args: argparse.Namespace) -> bool:
    return getattr(args, "confirm", False) or os.environ.get("CONFIRM") == "1"


def require_confirm(args: argparse.Namespace, action: str) -> None:
    if not confirmed(args):
        raise SystemExit(f"error: {action} requires CONFIRM=1 or --confirm")


def sh(script: Path, *args: str, check: bool = True, **env_kw: str) -> subprocess.CompletedProcess[str]:
    if not script.is_file():
        raise SystemExit(f"missing script: {script}")
    return subprocess.run(
        [str(script), *args],
        check=check,
        cwd=REPO,
        env=env(**env_kw),
    )


def lima(*args: str) -> None:
    subprocess.run(["limactl", *args], check=True, cwd=REPO)


def vm_exec(*args: str) -> None:
    sh(DEV_BIN / "vm-exec.sh", *args)


def maybe_sync(args: argparse.Namespace) -> None:
    if not getattr(args, "skip_sync", False) and os.environ.get("SKIP_SYNC") != "1":
        sh(DEV_BIN / "sync-repo.sh")


def cmd_help(args: argparse.Namespace) -> None:
    sh(DEV_BIN / "help.sh", getattr(args, "help_mode", "full"))


def cmd_doctor(_: argparse.Namespace) -> None:
    sh(DEV_BIN / "preflight.sh")


def cmd_version(_: argparse.Namespace) -> None:
    vm_exec(
        "bash", "-lc",
        "for f in /usr/local/bin/quark_d /usr/local/bin/quark "
        "/usr/local/bin/qkernel_d.bin /usr/local/bin/vdso.so; do "
        "[[ -e \"$f\" ]] && ls -lh \"$f\"; done; "
        "rustc --version 2>/dev/null || true",
    )


def cmd_vm(args: argparse.Namespace) -> None:
    action = args.action
    if action == "start":
        lima("start", LIMA)
    elif action == "stop":
        lima("stop", LIMA)
    elif action == "restart":
        lima("stop", LIMA)
        lima("start", LIMA)
    elif action == "shell":
        os.execvp("limactl", ["limactl", "shell", LIMA])
    elif action == "status":
        lima("list", LIMA)
    elif action == "create":
        lima("create", f"--name={LIMA}", "-y", str(LIMA_YAML))
    elif action == "delete":
        require_confirm(args, "vm delete")
        lima("delete", LIMA)
    elif action == "ssh-config":
        cfg = Path.home() / ".lima" / LIMA / "ssh.config"
        if cfg.is_file():
            print(cfg.read_text(), end="")
        else:
            print("VM not created yet")
    else:
        raise SystemExit(f"unknown vm action: {action}")


def cmd_sync(args: argparse.Namespace) -> None:
    if getattr(args, "watch", False):
        if not shutil.which("fswatch"):
            raise SystemExit("install fswatch: brew install fswatch")
        proc = subprocess.Popen(
            ["fswatch", "-o", "."],
            cwd=REPO,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        for _ in proc.stdout:
            sh(DEV_BIN / "sync-repo.sh")
    else:
        sh(DEV_BIN / "sync-repo.sh")


def cmd_build(args: argparse.Namespace) -> None:
    target = args.target
    maybe_sync(args)
    if target == "debug":
        vm_exec("make", "debug")
    elif target == "release":
        vm_exec("make", "release")
    elif target == "clean":
        vm_exec("make", "clean")
    elif target == "cleanall":
        require_confirm(args, "build cleanall")
        vm_exec("make", "cleanall")
    elif target == "qvisor":
        vm_exec("make", "-C", "qvisor", "debug")
    elif target == "qkernel":
        vm_exec("make", "-C", "qkernel", "debug")
    elif target == "vdso":
        vm_exec("make", "-C", "vdso")
    elif target == "cuda":
        vm_exec("make", "cuda_debug")
    elif target == "snp":
        vm_exec(
            "bash", "-lc",
            'arch=$(uname -m); '
            'if [[ "$arch" == "aarch64" ]]; then '
            'echo "warning: SNP build is x86-oriented; may fail on aarch64" >&2; fi; '
            "make snp_debug",
        )
    else:
        raise SystemExit(f"unknown build target: {target}")


def cmd_install(_: argparse.Namespace) -> None:
    vm_exec("sudo", "make", "install")
    vm_exec("sudo", "mkdir", "-p", "/var/log/quark")


def cmd_install_config(_: argparse.Namespace) -> None:
    vm_exec("sudo", "mkdir", "-p", "/etc/quark")
    vm_exec("sudo", "cp", "config.json", "/etc/quark/config.json")


def cmd_rebuild(args: argparse.Namespace) -> None:
    cmd_sync(argparse.Namespace(watch=False))
    cmd_build(argparse.Namespace(target="debug", skip_sync=True, confirm=args.confirm))
    cmd_install(args)
    vm_exec("sudo", "systemctl", "restart", "docker")
    print("✓ rebuild complete")


def cmd_run(args: argparse.Namespace) -> None:
    target = args.target
    flags = getattr(args, "docker_flags", None) or DOCKER_FLAGS
    runtime = getattr(args, "runtime", None) or RUNTIME

    if target == "hello":
        vm_exec("docker", "run", f"--runtime={runtime}", *flags.split(), "hello-world")
        print("✓ hello-world exited 0")
    elif target == "python":
        py_cmd = args.cmd or "import sys; print(sys.version); print(2+2)"
        vm_exec(
            "docker", "run", f"--runtime={runtime}", *flags.split(),
            "python:3.12-slim", "python", "-c", py_cmd,
        )
    elif target == "shell":
        vm_exec("docker", "run", f"--runtime={runtime}", "-it", "--rm", "ubuntu:24.04", "bash")
    elif target == "ubuntu":
        if not args.cmd:
            raise SystemExit("run ubuntu requires --cmd")
        vm_exec(
            "docker", "run", f"--runtime={runtime}", *flags.split(),
            "ubuntu:24.04", *shlex.split(args.cmd),
        )
    elif target == "busybox":
        vm_exec("docker", "run", f"--runtime={runtime}", "-it", "--rm", "busybox")
    elif target == "compare":
        sh(DEV_BIN / "test-startup.sh")
    elif target == "exec":
        docker_args = list(args.docker_args)
        if docker_args and docker_args[0] == "--":
            docker_args = docker_args[1:]
        if not docker_args:
            raise SystemExit("run exec requires docker arguments after --")
        vm_exec("docker", "run", f"--runtime={runtime}", *docker_args)
    else:
        raise SystemExit(f"unknown run target: {target}")


def cmd_logs(args: argparse.Namespace) -> None:
    action = args.action
    if action == "clear":
        require_confirm(args, "logs clear")
    lines = str(getattr(args, "lines", None) or os.environ.get("LINES", "200"))
    sh(DEV_BIN / "log-tail.sh", action, CONFIRM="1" if confirmed(args) else os.environ.get("CONFIRM", "0"), LINES=lines)


def cmd_config(args: argparse.Namespace) -> None:
    action = args.action
    if action == "shim":
        action = "shim-off" if args.shim == "off" else "shim-on"
    sh(DEV_BIN / "config-level.sh", action)


def cmd_docker(args: argparse.Namespace) -> None:
    action = args.action
    if action == "setup":
        sh(DEV_BIN / "docker-runtime.sh")
    elif action == "restart":
        vm_exec("sudo", "systemctl", "restart", "docker")
    elif action == "info":
        vm_exec("docker", "info")
    elif action == "pull":
        if not args.image and not os.environ.get("IMAGE"):
            raise SystemExit("docker pull requires --image or IMAGE=...")
        vm_exec("docker", "pull", args.image or os.environ["IMAGE"])
    else:
        raise SystemExit(f"unknown docker action: {action}")


def cmd_diag(args: argparse.Namespace) -> None:
    action = args.action
    if action == "all":
        for sub in ("vm", "kvm", "docker", "binaries", "mount", "config"):
            cmd_diag(argparse.Namespace(action=sub))
        print("✓ diag all passed")
        return
    if action == "vm":
        out = subprocess.run(
            ["limactl", "list", LIMA],
            capture_output=True, text=True, cwd=REPO,
        )
        lines = out.stdout.strip().splitlines()
        status = lines[1].split()[1] if len(lines) > 1 else "missing"
        if status != "Running":
            raise SystemExit(f"✗ Lima {LIMA} not running (status: {status})")
        print(f"✓ Lima {LIMA} Running")
        lima_yaml = Path.home() / ".lima" / LIMA / "lima.yaml"
        if lima_yaml.is_file():
            for line in lima_yaml.read_text().splitlines():
                if "nestedVirtualization" in line:
                    print(line.strip())
                    break
    elif action == "kvm":
        vm_exec("kvm-ok")
    elif action == "docker":
        sh(DEV_BIN / "diag-docker.sh")
    elif action == "binaries":
        sh(DEV_BIN / "diag-binaries.sh")
    elif action == "mount":
        sh(DEV_BIN / "diag-mount.sh")
    elif action == "config":
        vm_exec("python3", "-m", "json.tool", "/etc/quark/config.json")
        print("✓ config.json valid")
    elif action == "smoke":
        cmd_run(argparse.Namespace(target="hello", runtime=RUNTIME, docker_flags=DOCKER_FLAGS, cmd=None, docker_args=None))
    elif action == "aarch64":
        print("Quark aarch64 support is preliminary (see README.md).")
        print("PAN workaround patch: https://lists.sr.ht/~quark/QuarkContainer/patches/51839")
        vm_exec("uname", "-m")
    else:
        raise SystemExit(f"unknown diag action: {action}")


def cmd_watch(args: argparse.Namespace) -> None:
    target = args.target
    if target == "logs":
        sh(DEV_BIN / "log-tail.sh", "tail")
    elif target == "ps":
        sh(DEV_BIN / "watch-ps.sh")
    elif target == "build":
        if not shutil.which("entr"):
            raise SystemExit("install entr: brew install entr")
        find = subprocess.run(
            [
                "find", "qvisor", "qkernel", "vdso", "-type", "f", "(",
                "-name", "*.rs", "-o", "-name", "*.s", "-o", "-name", "*.cc",
                "-o", "-name", "makefile", "-o", "-name", "Makefile", ")",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["entr", "-c", sys.executable, str(REPO / "quark.py"), "rebuild"],
            input=find.stdout,
            text=True,
            cwd=REPO,
            check=True,
        )
    else:
        raise SystemExit(f"unknown watch target: {target}")


def cmd_test(args: argparse.Namespace) -> None:
    target = args.target
    runtime = getattr(args, "runtime", None) or RUNTIME
    flags = getattr(args, "docker_flags", None) or DOCKER_FLAGS
    if target == "smoke":
        cmd_run(argparse.Namespace(target="hello", runtime=runtime, docker_flags=flags, cmd=None, docker_args=None))
        cmd_run(argparse.Namespace(target="python", runtime=runtime, docker_flags=flags, cmd=None, docker_args=None))
    elif target == "startup":
        sh(DEV_BIN / "test-startup.sh")
    elif target == "memory":
        vm_exec("docker", "run", f"--runtime={runtime}", "-it", "--rm", "busybox")
    elif target == "rust":
        sh(DEV_BIN / "sync-repo.sh")
        vm_exec("make", "-C", "test/rust")
    elif target == "k8s":
        print("Not automated in v1.")
        print("See minikube_install.sh and doc/k8s_setup.md")
        print("Prereq: minikube with containerd inside Lima VM")
    elif target == "c":
        sh(DEV_BIN / "sync-repo.sh")
        test_name = args.test_name or os.environ.get("TEST", "fork")
        sh(DEV_BIN / "test-c.sh", test_name, runtime, flags)
    else:
        raise SystemExit(f"unknown test target: {target}")


def cmd_dev(args: argparse.Namespace) -> None:
    action = args.action
    if action in ("start", "up"):
        cmd_vm(argparse.Namespace(action="start"))
        cmd_diag(argparse.Namespace(action="all"))
    elif action in ("stop", "down"):
        cmd_vm(argparse.Namespace(action="stop"))
    elif action == "loop":
        cmd_config(argparse.Namespace(action="debug", shim="on"))
        cmd_rebuild(args)
        cmd_logs(argparse.Namespace(action="clear", confirm=True, lines=200))
        cmd_run(argparse.Namespace(target="hello", runtime=RUNTIME, docker_flags=DOCKER_FLAGS, cmd=None, docker_args=None))
        cmd_logs(argparse.Namespace(action="cat", confirm=False, lines=200))
    else:
        raise SystemExit(f"unknown dev action: {action}")


def cmd_bench(args: argparse.Namespace) -> None:
    if args.bench_cmd == "run":
        bench_args = [sys.executable, str(BENCH / "bench.py"), "run"]
        if args.profile:
            bench_args += ["--profile", args.profile]
        if args.suite:
            bench_args += ["--suite", args.suite]
        if args.runtime:
            bench_args += ["--runtime", args.runtime]
        if args.wave_size is not None:
            bench_args += ["--wave-size", str(args.wave_size)]
        if args.waves is not None:
            bench_args += ["--waves", str(args.waves)]
        subprocess.run(bench_args, check=True, cwd=REPO)
    elif args.bench_cmd == "setup":
        subprocess.run([sys.executable, str(BENCH / "bench.py"), "setup"], check=True, cwd=REPO)
    elif args.bench_cmd == "cleanup":
        subprocess.run([sys.executable, str(BENCH / "bench.py"), "cleanup"], check=True, cwd=REPO)
    elif args.bench_cmd == "results":
        subprocess.run([sys.executable, str(BENCH / "bench.py"), "results"], check=True, cwd=REPO)
    elif args.bench_cmd == "compare":
        if not args.files or len(args.files) < 2:
            raise SystemExit("bench compare needs two JSON files")
        subprocess.run(
            [sys.executable, str(BENCH / "compare.py"), *map(str, args.files)],
            check=True,
            cwd=REPO,
        )
    else:
        raise SystemExit(f"unknown bench command: {args.bench_cmd}")


def add_confirm(p: argparse.ArgumentParser) -> None:
    p.add_argument("--confirm", action="store_true", help="Confirm destructive action (or CONFIRM=1)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quark", description="Quark development CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("help", help="Show command groups").set_defaults(func=cmd_help, help_mode="full")
    sub.add_parser("quick", help="Quick start hints").set_defaults(func=cmd_help, help_mode="quick")
    sub.add_parser("doctor", help="Preflight checks").set_defaults(func=cmd_doctor)
    sub.add_parser("version", help="Show installed binary versions in VM").set_defaults(func=cmd_version)

    vm_p = sub.add_parser("vm", help="Lima VM control")
    vm_p.add_argument("action", choices=["start", "stop", "restart", "shell", "status", "create", "delete", "ssh-config"])
    add_confirm(vm_p)
    vm_p.set_defaults(func=cmd_vm)

    sync_p = sub.add_parser("sync", help="Sync repo to VM build tree")
    sync_p.add_argument("--watch", action="store_true", help="Auto-sync on file change (needs fswatch)")
    sync_p.set_defaults(func=cmd_sync)

    build_p = sub.add_parser("build", help="Build in VM")
    build_p.add_argument(
        "target",
        nargs="?",
        default="debug",
        choices=["debug", "release", "clean", "cleanall", "qvisor", "qkernel", "vdso", "cuda", "snp"],
    )
    build_p.add_argument("--skip-sync", action="store_true", help="Skip sync before build (or SKIP_SYNC=1)")
    add_confirm(build_p)
    build_p.set_defaults(func=cmd_build)

    sub.add_parser("install", help="Install binaries to /usr/local/bin").set_defaults(func=cmd_install)
    sub.add_parser("install-config", help="Copy config.json to /etc/quark").set_defaults(func=cmd_install_config)
    rebuild_p = sub.add_parser("rebuild", help="sync + build + install + docker restart")
    add_confirm(rebuild_p)
    rebuild_p.set_defaults(func=cmd_rebuild)

    run_p = sub.add_parser("run", help="Run containers in VM")
    run_p.add_argument(
        "target",
        choices=["hello", "python", "shell", "ubuntu", "busybox", "compare", "exec"],
    )
    run_p.add_argument("--cmd", help="Command for python/ubuntu")
    run_p.add_argument("--runtime", default=None)
    run_p.add_argument("--docker-flags", default=None)
    run_p.add_argument("docker_args", nargs=argparse.REMAINDER, help="Args for run exec (after --)")
    run_p.set_defaults(func=cmd_run)

    logs_p = sub.add_parser("logs", help="Quark log files in VM")
    logs_p.add_argument("action", choices=["tail", "cat", "clear"])
    logs_p.add_argument("--lines", type=int, default=None, help="Lines for cat (default 200)")
    add_confirm(logs_p)
    logs_p.set_defaults(func=cmd_logs)

    cfg_p = sub.add_parser("config", help="Quark config.json in VM")
    cfg_p.add_argument("action", choices=["show", "debug", "quiet", "shim"])
    cfg_p.add_argument("--shim", choices=["on", "off"], default="on", help="For config shim")
    cfg_p.set_defaults(func=cmd_config)

    docker_p = sub.add_parser("docker", help="Docker inside VM")
    docker_p.add_argument("action", choices=["setup", "restart", "info", "pull"])
    docker_p.add_argument("--image", default=None)
    docker_p.set_defaults(func=cmd_docker)

    diag_p = sub.add_parser("diag", help="Environment diagnostics")
    diag_p.add_argument(
        "action",
        choices=["all", "vm", "kvm", "docker", "binaries", "mount", "config", "smoke", "aarch64"],
    )
    diag_p.set_defaults(func=cmd_diag)

    watch_p = sub.add_parser("watch", help="Live monitoring / auto-rebuild")
    watch_p.add_argument("target", choices=["logs", "ps", "build"])
    watch_p.set_defaults(func=cmd_watch)

    test_p = sub.add_parser("test", help="Smoke and integration tests")
    test_p.add_argument("target", choices=["smoke", "startup", "memory", "rust", "k8s", "c"])
    test_p.add_argument("--test-name", default=None, help="C test name (default fork)")
    test_p.add_argument("--runtime", default=None)
    test_p.add_argument("--docker-flags", default=None)
    test_p.set_defaults(func=cmd_test)

    dev_p = sub.add_parser("dev", help="Dev workflows")
    dev_p.add_argument("action", choices=["start", "stop", "up", "down", "loop"])
    add_confirm(dev_p)
    dev_p.set_defaults(func=cmd_dev)

    bench_p = sub.add_parser("bench", help="Benchmark suite")
    bench_p.add_argument("bench_cmd", choices=["run", "setup", "cleanup", "results", "compare"])
    bench_p.add_argument("files", nargs="*", type=Path, help="JSON files for compare")
    bench_p.add_argument("--profile", choices=["dev", "stress"], default="dev")
    bench_p.add_argument("--suite", default="all")
    bench_p.add_argument("--runtime", default=None)
    bench_p.add_argument("--wave-size", type=int, default=None)
    bench_p.add_argument("--waves", type=int, default=None)
    bench_p.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    if not args.command:
        cmd_help(args)
        return 0
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
