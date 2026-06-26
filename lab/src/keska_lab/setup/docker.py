"""Docker runtime registration and image prep on the lab host."""

from __future__ import annotations

import json
import shlex
import textwrap

from keska_lab.config import LabConfig
from keska_lab.setup.base import SetupStep, StepResult
from keska_lab.remote import RemoteHost


def runtime_template(config: LabConfig) -> dict:
    del config  # runtimes are containerd shim types, not host paths
    return {
        "features": {
            "time-namespaces": False,
        },
        "runtimes": {
            "quark": {"runtimeType": "io.containerd.quark.v1"},
            "quark_d": {"runtimeType": "io.containerd.quarkd.v1"},
            "kata": {
                "runtimeType": "io.containerd.kata.v2",
            },
            # legacy alias
            "kata-runtime": {
                "runtimeType": "io.containerd.kata.v2",
            },
        },
    }


def ensure_docker_daemon_script() -> str:
    """Start dockerd if needed (safe to embed — never exit 0 early)."""
    return textwrap.dedent(
        """
        if ! sg docker -c "docker info >/dev/null 2>&1"; then
          sudo -n systemctl start docker 2>/dev/null || sudo -n service docker start 2>/dev/null || true
          for _i in $(seq 1 30); do
            sg docker -c "docker info >/dev/null 2>&1" && break
            sleep 1
          done
        fi
        if ! sg docker -c "docker info >/dev/null 2>&1"; then
          echo "docker info failed:" >&2
          sg docker -c "docker info 2>&1 | tail -15" >&2 || true
          exit 1
        fi
        """
    ).strip()


class DockerEnsureStep(SetupStep):
    """Ensure dockerd is running before mirror auth / bundle export."""

    name = "docker-ensure"

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        r = remote.sh(
            f"set -euo pipefail\n{ensure_docker_daemon_script()}",
            timeout=90,
            stream=stream,
        )
        if r.ok:
            return StepResult(self.name, True, "dockerd running")
        return StepResult(self.name, False, remote.format_failure(r) or "dockerd not running")


class DockerRuntimeStep(SetupStep):
    """Merge Quark + Kata runtimes into /etc/docker/daemon.json."""

    name = "docker-runtimes"

    def __init__(self, config: LabConfig | None = None):
        self.config = config or LabConfig.from_env()

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        template = json.dumps(runtime_template(self.config))
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            sudo -n mkdir -p /etc/docker
            if [ -f /etc/docker/daemon.json ]; then
              sudo -n cp /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%s)
            fi
            changed=$(sudo -n python3 - <<PY
            import json, pathlib
            template = json.loads({template!r})
            p = pathlib.Path("/etc/docker/daemon.json")
            if p.exists():
                data = json.loads(p.read_text())
            else:
                data = {{}}
            if "features" in template:
                data.setdefault("features", {{}}).update(template["features"])
            data.setdefault("runtimes", {{}}).update(template["runtimes"])
            new_text = json.dumps(data, indent=4, sort_keys=True) + "\\n"
            old_text = p.read_text() if p.exists() else ""
            if new_text == old_text:
                print("0")
            else:
                p.write_text(new_text)
                print("1")
            PY
            )
            if [ "$changed" = "1" ]; then
              # SIGHUP reloads daemon.json without a systemd restart — avoids
              # "Transaction is destructive (time-set.target)" on Docker 29+.
              if pid=$(pidof dockerd 2>/dev/null); then
                sudo -n kill -HUP "$pid"
              else
                sudo -n systemctl start docker
              fi
              sleep 2
            fi
            """
        ).strip()
        r = remote.sh(script, timeout=120, stream=stream)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        check = remote.docker_sh("docker info >/dev/null", timeout=60)
        if not check.ok:
            return StepResult(self.name, False, remote.format_failure(check))
        msg = "docker runtimes configured"
        if r.stdout.strip() == "0":
            msg = "docker runtimes unchanged (no reload needed)"
        return StepResult(self.name, True, msg)


class DockerPullStep(SetupStep):
    name = "docker-pull"

    def __init__(self, image: str, config: LabConfig | None = None):
        self.image = image
        self.config = config

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        from keska_lab.setup.image_registry import docker_pull_with_mirror_script

        registry = self.config.image_registry if self.config and self.config.image_registry else None
        r = remote.sh(
            docker_pull_with_mirror_script(self.image, registry),
            timeout=600,
            stream=stream,
        )
        return StepResult(self.name, r.ok, f"pulled {self.image}" if r.ok else remote.format_failure(r))


class DockerSanityStep(SetupStep):
    name = "docker-sanity"

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        r = remote.docker_sh("docker run --rm hello-world 2>&1 | tail -3", timeout=120)
        ok = r.ok and "Hello from Docker" in r.stdout
        return StepResult(self.name, ok, r.stdout.strip()[-200:] if r.stdout else r.stderr.strip())


class QuarkRuntimeCheckStep(SetupStep):
    name = "quark-runtime-check"

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        r = remote.docker_sh("docker info 2>/dev/null | grep -E 'quark|quark_d' || true", timeout=30)
        ok = "quark" in r.stdout.lower()
        return StepResult(
            self.name,
            ok,
            "quark runtime registered" if ok else "quark runtime not in docker info",
        )


class KataInstallStep(SetupStep):
    """Install Kata Containers static release (Ubuntu 24.04 has no distro package)."""

    name = "kata-install"

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        from keska_lab.setup.kata import ensure_kata

        try:
            msg = ensure_kata(remote, stream=stream)
            return StepResult(self.name, True, msg)
        except Exception as e:
            return StepResult(self.name, False, str(e))


class KataRuntimeCheckStep(SetupStep):
    """Legacy docker smoke — prefer KataCtrCheckStep in pipelines."""

    name = "kata-runtime-check"

    def __init__(self, config: LabConfig | None = None):
        self.config = config or LabConfig.from_env()

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        from keska_lab.setup.pipelines import KataCtrCheckStep

        return KataCtrCheckStep(self.config).run(remote, stream=stream)
