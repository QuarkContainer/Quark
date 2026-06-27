"""OCI bundle cache for direct Quark benchmarks (setup only, not timed)."""

from __future__ import annotations

import base64
import json
import re
import shlex
import textwrap

from keska_lab.config import LabConfig
from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupStep, StepResult
from keska_lab.setup.image_registry import (
    all_purge_refs,
    ctr_image_ref,
    ctr_pull_with_mirror_script,
    docker_pull_with_mirror_script,
)
from keska_lab.harness.workload import WorkloadSpec, get_workload

OCI_CONFIG = {
    "ociVersion": "1.0.2",
    "process": {
        "terminal": False,
        "user": {"uid": 0, "gid": 0},
        "args": ["/bin/sleep", "3600"],
        "env": ["PATH=/usr/sbin:/usr/bin:/sbin:/bin"],
        "cwd": "/",
    },
    "root": {"path": "rootfs"},
    "hostname": "lab",
    "mounts": [
        {"destination": "/proc", "type": "proc", "source": "proc"},
        {"destination": "/dev", "type": "tmpfs", "source": "tmpfs"},
        {"destination": "/sys", "type": "sysfs", "source": "sysfs"},
    ],
    "linux": {
        "namespaces": [
            {"type": "pid"},
            {"type": "network"},
            {"type": "ipc"},
            {"type": "uts"},
            {"type": "mount"},
        ]
    },
}


def image_slug(image: str) -> str:
    slug = image.split("/")[-1].split(":")[0]
    return re.sub(r"[^a-zA-Z0-9._-]", "_", slug) or "rootfs"


def bundle_dir(config: LabConfig, image: str) -> str:
    return f"{config.work_dir.rstrip('/')}/bundles/{image_slug(image)}"


def rootfs_markers_for_image(image: str) -> list[str]:
    """Paths under rootfs that must exist (any one suffices)."""
    if "postgres" in image.lower():
        return ["usr/local/bin/postgres"]
    return ["bin/busybox", "bin/sh"]


def bundle_template_dir(config: LabConfig, image: str) -> str:
    return f"{bundle_dir(config, image)}/rootfs.template"


def bundle_markers_ok_script(
    *, rootfs_var: str, image: str, use_return: bool = False
) -> str:
    """Shell fragment: success when at least one marker exists under ${rootfs_var}."""
    checks = " || ".join(
        f'[ -f "${{{rootfs_var}}}/{marker}" ]' for marker in rootfs_markers_for_image(image)
    )
    if use_return:
        return f"if {checks}; then return 0; else return 1; fi"
    return f"if {checks}; then exit 0; else exit 1; fi"


def verify_bundle_markers(remote: RemoteHost, bundle: str, image: str) -> None:
    script = bundle_markers_ok_script(rootfs_var="ROOTFS", image=image)
    r = remote.sh(
        f"ROOTFS={shlex.quote(bundle)}/rootfs; {script}",
        timeout=15,
        check=False,
    )
    if not r.ok:
        markers = ", ".join(rootfs_markers_for_image(image))
        raise RuntimeError(f"OCI bundle missing markers ({markers}) at {bundle}/rootfs")


def containerd_image_purge_script(*refs: str) -> str:
    """Remove image refs and orphaned snapshots across containerd namespaces."""
    uniq = list(dict.fromkeys(r for r in refs if r))
    refs_sh = " ".join(shlex.quote(r) for r in uniq)
    return textwrap.dedent(
        f"""
        set -euo pipefail
        refs=({refs_sh})
        purge_ns() {{
          ns="$1"
          for ref in "${{refs[@]}}"; do
            sudo -n ctr --namespace "$ns" images rm --sync "$ref" >/dev/null 2>&1 || true
            base="${{ref%%@*}}"
            sudo -n ctr --namespace "$ns" images rm --sync "$base" >/dev/null 2>&1 || true
          done
          for pass in $(seq 1 15); do
            changed=0
            for s in $(sudo -n ctr --namespace "$ns" snapshots ls 2>/dev/null | awk 'NR>1 {{print $1}}'); do
              kids=$(sudo -n ctr --namespace "$ns" snapshots ls | awk -v p="$s" '$2==p {{print $1}}')
              [ -z "$kids" ] && sudo -n ctr --namespace "$ns" snapshots rm "$s" >/dev/null 2>&1 && changed=1 || true
            done
            for s in $(sudo -n ctr --namespace "$ns" snapshots --snapshotter devmapper ls 2>/dev/null | awk 'NR>1 {{print $1}}'); do
              kids=$(sudo -n ctr --namespace "$ns" snapshots --snapshotter devmapper ls | awk -v p="$s" '$2==p {{print $1}}')
              [ -z "$kids" ] && sudo -n ctr --namespace "$ns" snapshots --snapshotter devmapper rm "$s" >/dev/null 2>&1 && changed=1 || true
            done
            [ "$changed" = 0 ] && break
          done
        }}
        for ns in default k8s.io moby; do
          purge_ns "$ns"
        done
        """
    ).strip()


def containerd_devmapper_metadata_reset_script() -> str:
    """Reset devmapper snapshotter side metadata (after --sync image rm)."""
    return textwrap.dedent(
        """
        set -euo pipefail
        sudo -n systemctl stop containerd 2>/dev/null || true
        sleep 1
        sudo -n rm -f /var/lib/containerd/devmapper/metadata.db
        sudo -n systemctl start containerd
        for i in $(seq 1 15); do
          sudo -n ctr plugins ls 2>/dev/null | grep -F devmapper | grep -q ' ok ' && break
          sleep 1
        done
        """
    ).strip()


def build_oci_config(
    *,
    args: tuple[str, ...] | None = None,
    env: tuple[str, ...] | None = None,
    extra_mounts: tuple[dict, ...] | None = None,
    user: tuple[int, int] | None = None,
) -> dict:
    cfg = json.loads(json.dumps(OCI_CONFIG))
    if args:
        cfg["process"]["args"] = list(args)
    if env:
        cfg["process"]["env"] = list(env)
    if user:
        cfg["process"]["user"] = {"uid": user[0], "gid": user[1]}
    if extra_mounts:
        destinations = {m["destination"] for m in cfg["mounts"]}
        for mount in extra_mounts:
            if mount["destination"] not in destinations:
                cfg["mounts"].append(dict(mount))
    return cfg


def refresh_bundle_rootfs_script(*, bundle_var: str, image: str) -> str:
    """Restore mutable rootfs from immutable template before quark create."""
    tmpl_var = f"${bundle_var}/rootfs.template"
    checks = " || ".join(
        f'[ -f "{tmpl_var}/{marker}" ]' for marker in rootfs_markers_for_image(image)
    )
    return textwrap.dedent(
        f"""
        if [ -d "${{{bundle_var}}}/rootfs.template" ] && ( {checks} ); then
          rm -rf "${{{bundle_var}}}/rootfs"
          cp -a "${{{bundle_var}}}/rootfs.template" "${{{bundle_var}}}/rootfs"
        fi
        """
    ).strip()


def ensure_bundle_script(config: LabConfig, image: str, *, oci: dict | None = None) -> str:
    bundle = bundle_dir(config, image)
    cfg_text = json.dumps(oci or OCI_CONFIG, indent=2) + "\n"
    cfg_b64 = base64.b64encode(cfg_text.encode()).decode()
    marker_ok = bundle_markers_ok_script(rootfs_var="TMPL", image=image, use_return=True)
    return textwrap.dedent(
        f"""
        set -euo pipefail
        BUNDLE={shlex.quote(bundle)}
        TMPL="$BUNDLE/rootfs.template"
        marker_ok() {{
          {marker_ok}
        }}
        if ! marker_ok; then
          rm -rf "$TMPL"
          mkdir -p "$TMPL"
          {docker_pull_with_mirror_script(image, config.image_registry or None)}
          cid=$(sg docker -c 'docker create {shlex.quote(image)}')
          if [ -z "$cid" ]; then
            echo "docker create failed for {shlex.quote(image)}" >&2
            exit 1
          fi
          sg docker -c "docker export \\"$cid\\" | tar -xC \\"$TMPL\\""
          sg docker -c "docker rm \\"$cid\\"" >/dev/null 2>&1 || true
          marker_ok || {{ echo "docker export missing rootfs markers in $TMPL" >&2; exit 1; }}
        fi
        rm -rf "$BUNDLE/rootfs"
        cp -a "$TMPL" "$BUNDLE/rootfs"
        marker_ok || {{ echo "rootfs copy missing markers" >&2; exit 1; }}
        echo {shlex.quote(cfg_b64)} | base64 -d | sudo -n tee "$BUNDLE/config.json" >/dev/null
        echo "$BUNDLE"
        """
    ).strip()


def ensure_oci_bundle(
    remote: RemoteHost,
    config: LabConfig,
    image: str,
    *,
    oci: dict | None = None,
) -> str:
    r = remote.sh(ensure_bundle_script(config, image, oci=oci), timeout=600)
    if not r.ok:
        raise RuntimeError(remote.format_failure(r) or "OCI bundle prepare failed")
    path = bundle_dir(config, image)
    check = remote.sh(
        f"test -f {shlex.quote(path)}/config.json && test -d {shlex.quote(path)}/rootfs && echo OK",
        timeout=30,
    )
    if not check.ok or "OK" not in check.stdout:
        tail = (r.stdout or r.stderr or check.stdout or check.stderr or "").strip()
        raise RuntimeError(f"OCI bundle not ready at {path}" + (f": {tail}" if tail else ""))
    verify_bundle_markers(remote, path, image)
    return path


def ensure_workload_bundle(remote: RemoteHost, config: LabConfig, workload: str | WorkloadSpec) -> str:
    spec = workload if isinstance(workload, WorkloadSpec) else get_workload(workload)
    oci = build_oci_config(
        args=spec.oci_args,
        env=spec.oci_env,
        extra_mounts=spec.oci_mounts or None,
        user=spec.oci_user,
    )
    path = ensure_oci_bundle(remote, config, spec.image, oci=oci)
    if spec.name == "postgres":
        _ensure_postgres_bundle_data(remote, path, config)
    return path


def postgres_data_template_dir(config: LabConfig) -> str:
    return f"{config.work_dir.rstrip('/')}/postgres-data-template"


def ensure_postgres_data_template(remote: RemoteHost, config: LabConfig) -> str:
    """Ensure pre-init postgres PGDATA exists on the lab host (shared by Quark and Kata)."""
    bundle = bundle_dir(config, "postgres:16-alpine")
    _ensure_postgres_bundle_data(remote, bundle, config)
    return postgres_data_template_dir(config)


def _ensure_postgres_bundle_data(remote: RemoteHost, bundle: str, config: LabConfig) -> None:
    template = postgres_data_template_dir(config)
    script = textwrap.dedent(
        f"""
        set -euo pipefail
        TMPL={shlex.quote(template)}
        if sudo test -f "$TMPL/PG_VERSION"; then
          exit 0
        fi
        mkdir -p "$TMPL"
        {docker_pull_with_mirror_script("postgres:16-alpine", config.image_registry or None)}
        cid=$(sg docker -c 'docker create -e POSTGRES_PASSWORD=bench -e POSTGRES_HOST_AUTH_METHOD=trust postgres:16-alpine')
        sg docker -c "docker start \\"$cid\\"" >/dev/null
        ready=0
        for i in $(seq 1 90); do
          if sg docker -c "docker exec \\"$cid\\" pg_isready -U postgres" >/dev/null 2>&1; then
            ready=1
            break
          fi
          sleep 1
        done
        test "$ready" = 1
        sg docker -c "docker exec --user postgres \\"$cid\\" pgbench -i -s1 -U postgres" >/dev/null
        sg docker -c "docker stop \\"$cid\\"" >/dev/null || true
        tmp=$(mktemp -d)
        sg docker -c "docker cp \\"$cid:/var/lib/postgresql/data/.\\" \\"$tmp/\\""
        sg docker -c "docker rm \\"$cid\\"" >/dev/null || true
        sudo rm -rf "$TMPL"
        sudo mkdir -p "$TMPL"
        sudo cp -a "$tmp/." "$TMPL/"
        rm -rf "$tmp"
        sudo chown -R 70:70 "$TMPL"
        sudo chmod 700 "$TMPL"
        sudo test -f "$TMPL/PG_VERSION"
        sudo rm -rf {shlex.quote(bundle)}/rootfs/var/lib/postgresql/data
        sudo find {shlex.quote(bundle)}/rootfs -maxdepth 3 -type d -name '*PostgreSQL*' -exec rm -rf {{}} + 2>/dev/null || true
        """
    ).strip()
    r = remote.sh(script, timeout=300)
    if not r.ok:
        raise RuntimeError(remote.format_failure(r) or "postgres data template init failed")


class EnsureWorkloadBundleStep(SetupStep):
    name = "workload-bundle"

    def __init__(self, config: LabConfig, workload: str):
        self.config = config
        self.workload = workload

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        try:
            path = ensure_workload_bundle(remote, self.config, self.workload)
            return StepResult(self.name, True, path)
        except Exception as e:
            return StepResult(self.name, False, str(e))


class EnsurePostgresDataTemplateStep(SetupStep):
    name = "postgres-data-template"

    def __init__(self, config: LabConfig):
        self.config = config

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        try:
            path = ensure_postgres_data_template(remote, self.config)
            return StepResult(self.name, True, path)
        except Exception as e:
            return StepResult(self.name, False, str(e))


class EnsureOciBundleStep(SetupStep):
    name = "oci-bundle"

    def __init__(self, config: LabConfig, image: str):
        self.config = config
        self.image = image

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        try:
            path = ensure_oci_bundle(remote, self.config, self.image)
            return StepResult(self.name, True, path)
        except Exception as e:
            return StepResult(self.name, False, str(e))


class CtrImagePullStep(SetupStep):
    """Pull bench image into containerd (for Kata ctr benchmarks)."""

    name = "ctr-pull"

    def __init__(self, image: str, config: LabConfig | None = None):
        self.image = image
        self.config = config

    def _pull_cmd(self, snap: str | None) -> str:
        registry = self.config.image_registry if self.config else None
        if not registry:
            registry = None
        return ctr_pull_with_mirror_script(self.image, registry, snapshotter=snap)

    def _purge_refs(self) -> list[str]:
        registry = self.config.image_registry if self.config else None
        return all_purge_refs(self.image, registry or None)

    def _snapshot_error(self, r) -> bool:
        text = f"{r.stdout}\n{r.stderr}".lower()
        return "snapshot" in text and ("does not exist" in text or "already exists" in text)

    def _docker_import(self, remote: RemoteHost, *, stream: bool) -> bool:
        registry = self.config.image_registry if self.config else None
        base = shlex.quote(f"docker.io/library/{image_slug(self.image)}")
        fb = remote.sh(
            f"{docker_pull_with_mirror_script(self.image, registry or None)} && "
            f"sg docker -c 'docker save {shlex.quote(self.image)}' | "
            f"sudo -n ctr images import --base-name {base} --digests -",
            timeout=900,
            stream=stream,
        )
        return fb.ok

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        ref = ctr_image_ref(self.image)
        snap = self.config.kata_snapshotter if self.config else None
        purge = self._purge_refs()
        pull_cmd = self._pull_cmd(snap)

        if snap:
            remote.sh(containerd_image_purge_script(*purge), timeout=120, stream=stream)

        r = remote.sh(pull_cmd, timeout=600, stream=stream)
        if not r.ok and snap and self._snapshot_error(r):
            remote.sh(containerd_image_purge_script(*purge), timeout=120, stream=stream)
            remote.sh(containerd_devmapper_metadata_reset_script(), timeout=180, stream=stream)
            r = remote.sh(pull_cmd, timeout=600, stream=stream)

        if r.ok:
            return StepResult(self.name, True, f"pulled {ref}")

        if snap:
            remote.sh(containerd_image_purge_script(*purge), timeout=120, stream=stream)
            if self._docker_import(remote, stream=stream):
                r = remote.sh(pull_cmd, timeout=600, stream=stream)
                if r.ok:
                    return StepResult(self.name, True, f"imported + devmapper unpack {ref}")
            return StepResult(self.name, False, remote.format_failure(r))

        if self._docker_import(remote, stream=stream):
            return StepResult(self.name, True, f"imported {ref} via docker")
        check = remote.sh(
            f"sudo -n ctr images ls name | grep -F {shlex.quote(ref.split(':')[0])} || true",
            timeout=30,
        )
        if check.stdout.strip():
            return StepResult(self.name, True, f"image {ref} ready")
        return StepResult(self.name, False, remote.format_failure(r))
