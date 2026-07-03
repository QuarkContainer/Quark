"""Enable containerd CRI + Quark shim runtimes (lab host only)."""

from __future__ import annotations

import textwrap

from keska_lab.cri.lifecycle import cri_lifecycle_smoke_script
from keska_lab.harness.network import crictl_config_script
from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupStep, StepResult

MARKER = "keska-lab cri"
OLD_MARKERS = ("keska-lab quark runtimes", MARKER, "keska-lab devmapper")
CRICTL_VERSION = "v1.31.1"
CRICTL_URL = (
    f"https://github.com/kubernetes-sigs/cri-tools/releases/download/"
    f"{CRICTL_VERSION}/crictl-{CRICTL_VERSION}-linux-amd64.tar.gz"
)

# Clean containerd 2.x config (cri.v1). Devmapper is added by kata_firecracker when needed.
CONTAINERD_PATCH_PY = r'''
import re
import shutil
import subprocess
from pathlib import Path

OLD_MARKERS = __OLD_MARKERS__

cfg_path = Path("/etc/containerd/config.toml")
backup = cfg_path.with_suffix(".toml.bak.keska")
if cfg_path.exists() and not backup.exists():
    shutil.copy2(cfg_path, backup)

base = subprocess.check_output(["containerd", "config", "default"], text=True)

# CNI paths (lab)
base = base.replace(
    "bin_dir = ''",
    "bin_dir = '/opt/cni/bin'",
    1,
)
base = base.replace(
    "conf_dir = ''",
    "conf_dir = '/etc/cni/net.d'",
    1,
)

# Default runtime + Quark/Kata shims
base = base.replace(
    "default_runtime_name = 'runc'",
    "default_runtime_name = 'quark'",
)

extra_runtimes = """
      [plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.quark]
        runtime_type = 'io.containerd.quark.v1'
        sandboxer = 'podsandbox'
      [plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.quarkd]
        runtime_type = 'io.containerd.quarkd.v1'
        sandboxer = 'podsandbox'
      [plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.kata]
        runtime_type = 'io.containerd.kata.v2'
        sandboxer = 'podsandbox'
        snapshotter = 'devmapper'
"""
if "runtimes.quark" not in base:
    runc_end = base.find("\n\n", base.find("runtimes.runc.options"))
    if runc_end == -1:
        raise SystemExit("could not locate end of runc runtime block")
    base = base[:runc_end] + "\n" + extra_runtimes + base[runc_end:]
else:
    for name in ("quark", "quarkd", "kata"):
        block = f"runtimes.{name}]"
        if block not in base:
            continue
        start = base.find(block)
        end = base.find("\n      [", start + 1)
        if end == -1:
            end = base.find("\n\n", start + 1)
        section = base[start:end if end != -1 else len(base)]
        if "sandboxer" not in section:
            insert_at = base.find("\n", start) + 1
            base = base[:insert_at] + "        sandboxer = 'podsandbox'\n" + base[insert_at:]

# Pause image available on lab
base = re.sub(
    r"sandbox = 'registry\.k8s\.io/pause:[^']+'",
    "sandbox = 'registry.k8s.io/pause:3.8'",
    base,
)

# containerd 2.x Transfer API needs explicit unpack platforms for CRI + ctr devmapper
anchor = "[plugins.'io.containerd.transfer.v1.local']"
idx = base.find(anchor)
if idx == -1:
    raise SystemExit("containerd default config missing transfer.v1.local block")
end = base.find("\\n\\n", idx)
if end == -1:
    end = len(base)
transfer_tail = base[idx:end]
extra_unpack = ""
if "snapshotter = 'overlayfs'" not in transfer_tail:
    extra_unpack += """
    [[plugins.'io.containerd.transfer.v1.local'.unpack_config]]
      platform = 'linux/amd64'
      snapshotter = 'overlayfs'
"""
if not re.search(
    r"\\[\\[plugins\\.'io\\.containerd\\.transfer\\.v1\\.local'\\.unpack_config\\]\\][\\s\\S]*?snapshotter = 'devmapper'",
    base,
):
    extra_unpack += """
    [[plugins.'io.containerd.transfer.v1.local'.unpack_config]]
      platform = 'linux/amd64'
      snapshotter = 'devmapper'
"""
if extra_unpack:
    base = base[:end] + extra_unpack + base[end:]

cfg_path.write_text(base)
print("wrote containerd config (containerd 2.x cri.v1)")
'''.replace("__OLD_MARKERS__", repr(OLD_MARKERS))


def containerd_restart_script() -> str:
    return textwrap.dedent(
        """
        set -euo pipefail
        sudo -n systemctl daemon-reload
        for id in $(sudo crictl ps -q 2>/dev/null || true); do
          sudo crictl stop "$id" 2>/dev/null || true
          sudo crictl rm "$id" 2>/dev/null || true
        done
        for id in $(sudo crictl pods -q 2>/dev/null || true); do
          sudo crictl stopp "$id" 2>/dev/null || true
          sudo crictl rmp "$id" 2>/dev/null || true
        done
        sudo -n systemctl stop containerd 2>/dev/null || true
        sleep 2
        sudo -n pkill -9 containerd-shim 2>/dev/null || true
        sudo -n systemctl start containerd || true
        for i in $(seq 1 60); do
          if sudo -n crictl info >/dev/null 2>&1; then
            break
          fi
          sleep 2
        done
        sudo -n crictl info >/dev/null
        sudo -n chmod 666 /run/containerd/containerd.sock
        """
    ).strip()


def containerd_cri_config_script() -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        sudo -n python3 <<'PY'
{CONTAINERD_PATCH_PY}
PY
        {containerd_restart_script()}
        sudo -n ctr -n k8s.io images pull registry.k8s.io/pause:3.8 2>/dev/null | tail -1 || true
        sudo -n ctr -n k8s.io images pull docker.io/library/busybox:latest 2>/dev/null | tail -1 || true
        echo "containerd cri.v1 configured"
        """
    ).strip()


def cri_smoke_script() -> str:
    return textwrap.dedent(
        """
        set -euo pipefail
        sudo -n crictl info >/dev/null
        echo "crictl ok"
        """
    ).strip()


def cri_lifecycle_setup_smoke_script(*, runtime_handler: str = "") -> str:
    """L2-equivalent smoke for ContainerdCriStep — runp → create → start → exec → teardown."""
    return cri_lifecycle_smoke_script(runtime_handler=runtime_handler, pod_name="keska-cri-setup")


def cri_stats_smoke_script(*, memory_min_bytes: int = 1) -> str:
    """H2: crictl stats via quark shim + cgroup reader."""
    return textwrap.dedent(
        f"""
        set -euo pipefail
        LOG=/tmp/keska-cri-stats-logs
        mkdir -p "$LOG"
        WD=/tmp/keska-cri-stats-$$
        mkdir -p "$WD"
        CID=""
        POD=""
        cleanup() {{
          if [ -n "$CID" ]; then
            sudo crictl stop -t 30 "$CID" 2>/dev/null || true
            sudo crictl rm "$CID" 2>/dev/null || true
          fi
          if [ -n "$POD" ]; then
            sudo crictl stopp "$POD" 2>/dev/null || true
            sudo crictl rmp -f "$POD" 2>/dev/null || true
          fi
          rm -rf "$WD"
        }}
        trap cleanup EXIT INT TERM
        cat >"$WD/pod.json" <<JSON
{{"metadata":{{"name":"keska-cri-stats","uid":"keska-cri-stats-uid","namespace":"default"}},"log_directory":"$LOG","linux":{{}}}}
JSON
        cat >"$WD/container.json" <<'JSON'
{{"metadata":{{"name":"keska-cri-stats-c","namespace":"default"}},"image":{{"image":"docker.io/library/busybox:latest"}},"command":["/bin/sleep","600"],"log_path":"stats.log","linux":{{"resources":{{"memory_limit_in_bytes":67108864}}}}}}
JSON
        POD=$(sudo crictl runp "$WD/pod.json")
        CID=$(sudo crictl create --no-pull "$POD" "$WD/container.json" "$WD/pod.json")
        sudo crictl start "$CID"
        POD=$(sudo crictl pods -q --name keska-cri-stats | head -1)
        sleep 4
        sudo crictl stats "$CID" | python3 -c "
import re, sys
text = sys.stdin.read()
# table row: id name cpu mem ...
m = re.search(r'\\n[0-9a-f]+\\s+\\S+\\s+[\\d.]+\\s+([\\d.]+)(kB|MB|GB|B)', text)
if not m:
    raise SystemExit(f'stats parse failed: {{text!r}}')
val, unit = float(m.group(1)), m.group(2)
mult = {{'B': 1, 'kB': 1024, 'MB': 1024**2, 'GB': 1024**3}}
usage = int(val * mult.get(unit, 1))
print(f'memory_usage={{usage}}')
if usage < {memory_min_bytes}:
    raise SystemExit('stats memory usage too low')
print('H2 PASS: crictl stats')
"
        """
    ).strip()


class CrictlInstallStep(SetupStep):
    name = "crictl-install"

    def run(self, remote: RemoteHost) -> StepResult:
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            if ! command -v crictl >/dev/null 2>&1; then
              tmp=$(mktemp -d)
              curl -fsSL {CRICTL_URL!r} -o "$tmp/crictl.tgz"
              sudo -n tar -C /usr/local/bin -xzf "$tmp/crictl.tgz" crictl
              rm -rf "$tmp"
            fi
            {crictl_config_script()}
            crictl --version | head -1
            """
        ).strip()
        r = remote.sh(script, timeout=120)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        ver = r.stdout.strip().splitlines()[-1]
        return StepResult(self.name, True, ver)


class ContainerdCriStep(SetupStep):
    name = "containerd-cri"

    def run(self, remote: RemoteHost) -> StepResult:
        r = remote.sh(containerd_cri_config_script(), timeout=300)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        smoke = remote.sh(cri_smoke_script(), timeout=60)
        if not smoke.ok:
            return StepResult(self.name, False, remote.format_failure(smoke))
        run_smoke = remote.sh(cri_lifecycle_setup_smoke_script(), timeout=300)
        if not run_smoke.ok:
            return StepResult(self.name, False, remote.format_failure(run_smoke))
        msg = run_smoke.stdout.strip().splitlines()[-1]
        return StepResult(self.name, True, msg or "cri enabled")


class QuarkCriStatsStep(SetupStep):
    """H2 integration: crictl stats on a quark CRI container."""

    name = "quark-cri-stats"

    def run(self, remote: RemoteHost) -> StepResult:
        from keska_lab.setup.quark_config import deploy_config_script, cri_bench_config_json

        cfg = cri_bench_config_json()
        r = remote.sh(deploy_config_script(cfg), timeout=60)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        r = remote.sh(cri_stats_smoke_script(), timeout=180)
        if not r.ok:
            return StepResult(self.name, False, remote.format_failure(r))
        line = r.stdout.strip().splitlines()[-1]
        return StepResult(self.name, True, line)
