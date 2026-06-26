"""Kata + Firecracker setup — requires containerd devmapper snapshotter."""

from __future__ import annotations

import textwrap

from keska_lab.config import LabConfig
from keska_lab.setup.image_registry import DEFAULT_IMAGE_REGISTRY, ctr_pull_with_mirror_script
from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupStep, StepResult

KATA_ROOT = "/opt/kata"
DEVMAPPER_DATA_DIR = "/var/lib/containerd/devmapper"
DEVMAPPER_POOL = "devpool"


def configure_kata_hypervisor_script(hypervisor: str = "firecracker") -> str:
    """Point /etc/kata-containers/configuration.toml at the chosen hypervisor config."""
    configs = {
        "firecracker": "configuration-fc.toml",
        "cloud-hypervisor": "configuration-clh.toml",
        "qemu": "configuration-qemu.toml",
    }
    cfg_file = configs.get(hypervisor, configs["firecracker"])
    return textwrap.dedent(
        f"""
        set -euo pipefail
        test -f {KATA_ROOT}/share/defaults/kata-containers/{cfg_file}
        sudo -n ln -sf {cfg_file} /etc/kata-containers/configuration.toml
        echo "kata hypervisor: {hypervisor} ({cfg_file})"
        """
    ).strip()


def devmapper_pool_script(image_registry: str | None = DEFAULT_IMAGE_REGISTRY) -> str:
    """Create devmapper thin pool (once) — required for Kata + Firecracker."""
    pull = ctr_pull_with_mirror_script(
        "busybox", image_registry or DEFAULT_IMAGE_REGISTRY, snapshotter="devmapper"
    )
    return textwrap.dedent(
        f"""
        set -euo pipefail
        DATA_DIR={DEVMAPPER_DATA_DIR}
        POOL_NAME={DEVMAPPER_POOL}

        pool_exists() {{
          sudo -n dmsetup info "$POOL_NAME" >/dev/null 2>&1
        }}

        health_ok() {{
          pool_exists || return 1
          id=keska-dm-health-$RANDOM
          if timeout 45 sudo -n ctr run --rm --runtime io.containerd.kata.v2 --snapshotter devmapper \\
            docker.io/library/busybox:latest "$id" /bin/true >/dev/null 2>&1; then
            return 0
          fi
          sudo -n ctr images rm --sync docker.io/library/busybox:latest >/dev/null 2>&1 || true
          if ! {pull}; then
            echo "devmapper health: busybox pull failed" >&2
            return 1
          fi
          id=keska-dm-health-$RANDOM
          if ! timeout 45 sudo -n ctr run --rm --runtime io.containerd.kata.v2 --snapshotter devmapper \\
            docker.io/library/busybox:latest "$id" /bin/true >/tmp/keska-dm-health-run.log 2>&1; then
            echo "devmapper health: kata run failed: $(tail -1 /tmp/keska-dm-health-run.log)" >&2
            return 1
          fi
          return 0
        }}

        clear_stale_kata() {{
          sudo -n killall -9 firecracker >/dev/null 2>&1 || true
          for ns in default k8s.io; do
            ids=$(sudo -n ctr --namespace "$ns" containers ls -q 2>/dev/null | grep -E '^keska-' || true)
            for id in $ids; do
              timeout 3 sudo -n ctr --namespace "$ns" task kill -s SIGKILL "$id" >/dev/null 2>&1 || true
              timeout 3 sudo -n ctr --namespace "$ns" containers rm "$id" >/dev/null 2>&1 || true
            done
          done
          sleep 1
        }}

        force_remove_pool() {{
          echo "releasing devmapper pool"
          sudo -n systemctl stop containerd 2>/dev/null || true
          sleep 2
          for ns in default k8s.io; do
            ids=$(sudo -n ctr --namespace "$ns" containers ls -q 2>/dev/null || true)
            for id in $ids; do
              sudo -n ctr --namespace "$ns" task kill -s SIGKILL "$id" >/dev/null 2>&1 || true
              sudo -n ctr --namespace "$ns" containers rm "$id" >/dev/null 2>&1 || true
            done
            imgs=$(sudo -n ctr --namespace "$ns" images ls -q 2>/dev/null || true)
            for img in $imgs; do
              sudo -n ctr --namespace "$ns" images rm --sync "$img" >/dev/null 2>&1 || true
            done
          done
          for dev in $(sudo -n dmsetup ls 2>/dev/null | awk '{{print $1}}' | tac); do
            [ "$dev" = "$POOL_NAME" ] && continue
            sudo -n dmsetup remove "$dev" >/dev/null 2>&1 || true
          done
          sudo -n dmsetup remove "$POOL_NAME" >/dev/null 2>&1 || true
          for f in "$DATA_DIR/data" "$DATA_DIR/meta"; do
            sudo -n losetup -j "$f" 2>/dev/null | cut -d: -f1 | while read -r loop; do
              [ -n "$loop" ] && sudo -n losetup -d "$loop" >/dev/null 2>&1 || true
            done
          done
          sudo -n systemctl restart containerd 2>/dev/null || true
          sleep 4
        }}

        wait_devmapper_plugin() {{
          for i in $(seq 1 15); do
            if sudo -n ctr plugins ls 2>/dev/null | grep -F devmapper | grep -q ' ok '; then
              return 0
            fi
            sleep 1
          done
          return 1
        }}

        create_pool_from_files() {{
          force_remove_pool
          DATA_DEV=$(sudo -n losetup --find --show "$DATA_DIR/data")
          META_DEV=$(sudo -n losetup --find --show "$DATA_DIR/meta")
          SECTOR_SIZE=512
          DATA_SIZE=$(sudo -n blockdev --getsize64 -q "$DATA_DEV")
          LENGTH_IN_SECTORS=$(( DATA_SIZE / SECTOR_SIZE ))
          if ! sudo -n dmsetup create "$POOL_NAME" \\
              --table "0 ${{LENGTH_IN_SECTORS}} thin-pool ${{META_DEV}} ${{DATA_DEV}} 128 32768"; then
            force_remove_pool
            sudo -n dmsetup create "$POOL_NAME" \\
              --table "0 ${{LENGTH_IN_SECTORS}} thin-pool ${{META_DEV}} ${{DATA_DEV}} 128 32768"
          fi
          sudo -n systemctl restart containerd
          wait_devmapper_plugin
          echo "devpool created"
        }}

        reset_pool() {{
          echo "resetting devmapper pool"
          force_remove_pool
          sudo -n rm -f "$DATA_DIR/data" "$DATA_DIR/meta" "$DATA_DIR/devpool.db" "$DATA_DIR/metadata.db"
          sudo -n mkdir -p "$DATA_DIR"
          sudo -n truncate -s 50G "$DATA_DIR/data"
          sudo -n truncate -s 2G "$DATA_DIR/meta"
          create_pool_from_files
        }}

        sudo -n mkdir -p "$DATA_DIR"
        echo "checking devmapper pool..."
        if pool_exists && health_ok; then
          echo "devpool already active"
        elif pool_exists; then
          echo "devpool unhealthy — clearing stale kata sandboxes"
          clear_stale_kata
          if health_ok; then
            echo "devpool recovered after cleanup"
          elif [ -f "$DATA_DIR/data" ] && [ -f "$DATA_DIR/meta" ]; then
            create_pool_from_files
          else
            reset_pool
          fi
        elif [ -f "$DATA_DIR/data" ] && [ -f "$DATA_DIR/meta" ]; then
          create_pool_from_files
        else
          reset_pool
        fi
        health_ok || {{
          echo "devmapper pool health check failed" >&2
          exit 1
        }}
        echo "devpool ready"
        """
    ).strip()


def containerd_devmapper_config_script() -> str:
    """Enable devmapper snapshotter in containerd (patch main config — conf.d merge is unreliable)."""
    marker = "keska-lab devmapper"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        out=$(sudo -n python3 <<'PY'
import re
from pathlib import Path

marker = {marker!r}
cfg_path = Path("/etc/containerd/config.toml")
text = cfg_path.read_text()
changed = False

if marker not in text:
    text += f"\\n# {{marker}}\\n"
    changed = True

if not re.search(r'pool_name\\s*=\\s*["\\']devpool["\\']', text):
    text += f'''
[plugins."io.containerd.snapshotter.v1.devmapper"]
  root_path = "{DEVMAPPER_DATA_DIR}"
  pool_name = "{DEVMAPPER_POOL}"
  base_image_size = "512MB"
  discard_blocks = true
'''
    changed = True
else:
    new_text, n = re.subn(
        r'base_image_size\\s*=\\s*"[^"]+"',
        'base_image_size = "512MB"',
        text,
        count=1,
    )
    if n:
        text = new_text
        changed = True

if 'snapshotter = "devmapper"' not in text and "snapshotter = 'devmapper'" not in text:
    text += '''
[[plugins."io.containerd.transfer.v1.local".unpack_config]]
  platform = "linux/amd64"
  snapshotter = "devmapper"
'''
    changed = True

if changed:
    cfg_path.write_text(text)
    print("changed")
else:
    print("unchanged")
PY
        )
        echo "$out"
        if [ "$out" = "changed" ]; then
          sudo -n systemctl restart containerd
          sleep 4
        fi
        sudo -n ctr plugins ls | grep -F 'devmapper' | grep -q ' ok '
        """
    ).strip()


def ensure_kata_firecracker(
    remote: RemoteHost,
    *,
    image_registry: str | None = None,
    stream: bool = False,
) -> str:
    registry = image_registry if image_registry is not None else remote.config.image_registry
    steps = [configure_kata_hypervisor_script("firecracker")]
    devmapper_ok = remote.sh(
        "sudo -n ctr plugins ls 2>/dev/null | grep -F devmapper | grep -q ' ok ' && "
        "test -f /etc/containerd/config.toml",
        timeout=30,
        stream=stream,
    )
    if not devmapper_ok.ok:
        steps.append(containerd_devmapper_config_script())
    steps.append(devmapper_pool_script(registry))
    timeouts = (60, 180, 600) if len(steps) == 3 else (60, 600)
    for script, timeout in zip(steps, timeouts):
        r = remote.sh(script, timeout=timeout, stream=stream)
        if not r.ok:
            raise RuntimeError(remote.format_failure(r))
    return "kata firecracker + devmapper ready"


class KataFirecrackerStep(SetupStep):
    """Configure Kata to use Firecracker (devmapper snapshotter required)."""

    name = "kata-firecracker"

    def __init__(self, config: LabConfig | None = None):
        self.config = config

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        try:
            registry = self.config.image_registry if self.config else remote.config.image_registry
            msg = ensure_kata_firecracker(remote, image_registry=registry, stream=stream)
            return StepResult(self.name, True, msg)
        except Exception as e:
            return StepResult(self.name, False, str(e))


class KataHypervisorStep(SetupStep):
    """Select Kata hypervisor config (firecracker, cloud-hypervisor, qemu)."""

    name = "kata-hypervisor"

    def __init__(self, config: LabConfig):
        self.config = config
        self.hypervisor = config.kata_hypervisor

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        if self.hypervisor == "firecracker":
            return KataFirecrackerStep(self.config).run(remote, stream=stream)
        script = configure_kata_hypervisor_script(self.hypervisor)
        r = remote.sh(script, timeout=60, stream=stream)
        ok = r.ok
        msg = r.stdout.strip().splitlines()[-1] if ok else remote.format_failure(r)
        return StepResult(self.name, ok, msg)
