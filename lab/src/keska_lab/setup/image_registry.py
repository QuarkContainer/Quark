"""Keska Artifact Registry mirror — try before Docker Hub."""

from __future__ import annotations

import shlex
import textwrap
from typing import TYPE_CHECKING

from keska_lab.config import LabConfig
from keska_lab.setup.base import SetupStep, StepResult
from keska_lab.setup.docker import ensure_docker_daemon_script

if TYPE_CHECKING:
    from keska_lab.remote import RemoteHost

DEFAULT_IMAGE_REGISTRY = "europe-north1-docker.pkg.dev/keska-devops/base-images"
GCLOUD_AUTH_LOGIN_CMD = "gcloud auth login --no-launch-browser"


def ctr_image_ref(image: str) -> str:
    """Canonical containerd/docker reference (docker.io normalized)."""
    if "/" in image:
        return image if ":" in image.split("/")[-1] else f"{image}:latest"
    if ":" in image:
        return f"docker.io/library/{image}"
    return f"docker.io/library/{image}:latest"


def cri_image_ref(image: str) -> str:
    """Reference for crictl/CRI (docker.io prefix for short namespaced images)."""
    ref = ctr_image_ref(image)
    head = ref.split("/")[0]
    if "/" in ref and "." not in head and head not in ("localhost", "127.0.0.1"):
        return f"docker.io/{ref}"
    return ref


def mirror_image_ref(image: str, registry: str) -> str:
    """Map a workload short name to the Keska base-images registry."""
    registry = registry.rstrip("/")
    if not registry:
        return ctr_image_ref(image)
    head = image.split("/")[0]
    if "/" in image and ("." in head or head in ("localhost", "127.0.0.1")):
        return image if ":" in image.split("/")[-1] else f"{image}:latest"
    if "/" in image:
        return image if ":" in image.split("/")[-1] else f"{registry}/{image}:latest"
    if ":" in image:
        return f"{registry}/{image}"
    return f"{registry}/{image}:latest"


def image_pull_refs(image: str, registry: str | None) -> list[str]:
    """Pull sources in order: Keska mirror first, then upstream."""
    upstream = ctr_image_ref(image)
    if not registry:
        return [upstream]
    mirror = mirror_image_ref(image, registry)
    if mirror == upstream:
        return [upstream]
    return [mirror, upstream]


def registry_host(registry: str) -> str:
    """Hostname for docker credential helper (europe-north1-docker.pkg.dev)."""
    return registry.rstrip("/").split("/")[0]


def registry_probe_ref(registry: str) -> str:
    """Small image used to verify mirror auth."""
    return mirror_image_ref("busybox", registry)


def lab_path_setup_script() -> str:
    """Ensure gcloud is on PATH for non-interactive SSH sessions."""
    return textwrap.dedent(
        """
        for _g in "$HOME/google-cloud-sdk/bin" /usr/local/google-cloud-sdk/bin /usr/local/bin /usr/bin; do
          if [ -x "$_g/gcloud" ]; then
            export PATH="$_g:$PATH"
            break
          fi
        done
        """
    ).strip()


def gcloud_auth_login_script() -> str:
    """Interactive GCP login on the lab host (URL in terminal, paste code at prompt)."""
    return f"{lab_path_setup_script()}\n{GCLOUD_AUTH_LOGIN_CMD}"


def gcloud_docker_login_script(host: str) -> str:
    """Authenticate docker to GCP Artifact Registry via gcloud access token."""
    host_q = shlex.quote(host)
    return textwrap.dedent(
        f"""
        if command -v gcloud >/dev/null 2>&1; then
          gcloud auth configure-docker {host_q} --quiet >/dev/null 2>&1 || true
          if token=$(gcloud auth print-access-token 2>/dev/null); then
            echo "$token" | sg docker -c "docker login -u oauth2accesstoken --password-stdin https://{host_q}" >/dev/null 2>&1 || true
          fi
        fi
        """
    ).strip()


def docker_auth_preamble(registry: str | None) -> str:
    parts = [lab_path_setup_script(), ensure_docker_daemon_script()]
    if registry:
        parts.append(gcloud_docker_login_script(registry_host(registry)))
    return "\n".join(parts)


def ctr_mirror_pull_auth_shell(registry: str | None) -> str:
    """Shell helper: gcloud token flags for ctr pull of Keska mirror refs."""
    if not registry:
        return textwrap.dedent(
            """
            ctr_pull_ref() {
              sudo -n ctr images pull "$@"
            }
            """
        ).strip()
    host = registry_host(registry)
    host_q = shlex.quote(host)
    return textwrap.dedent(
        f"""
        ctr_pull_ref() {{
          local ref=$1
          shift
          local auth=()
          if [[ "$ref" == *{host_q}* ]]; then
            {lab_path_setup_script()}
            if token=$(gcloud auth print-access-token 2>/dev/null); then
              auth=(--user oauth2accesstoken --secret "$token")
            fi
          fi
          sudo -n ctr images pull "${{auth[@]}}" "$@" "$ref"
        }}
        """
    ).strip()


def configure_gcp_docker_script(registry: str) -> str:
    host = registry_host(registry)
    return textwrap.dedent(
        f"""
        set -euo pipefail
        {docker_auth_preamble(registry)}
        command -v gcloud >/dev/null
        gcloud auth configure-docker {shlex.quote(host)} --quiet
        """
    ).strip()


def check_registry_auth_ctr_script(registry: str) -> str:
    """Verify mirror access via ctr + gcloud token (no dockerd)."""
    probe = registry_probe_ref(registry)
    return textwrap.dedent(
        f"""
        set -euo pipefail
        {lab_path_setup_script()}
        probe={shlex.quote(probe)}
        command -v gcloud >/dev/null
        token=$(gcloud auth print-access-token)
        sudo -n ctr images pull --user oauth2accesstoken --secret "$token" --platform linux/amd64 "$probe"
        sudo -n ctr images rm "$probe" >/dev/null 2>&1 || true
        """
    ).strip()


def check_registry_auth_script(registry: str) -> str:
    """Return 0 when docker can pull the mirror probe image."""
    probe = registry_probe_ref(registry)
    return textwrap.dedent(
        f"""
        set -euo pipefail
        {docker_auth_preamble(registry)}
        probe={shlex.quote(probe)}
        sg docker -c "docker pull \\"$probe\\"" >/dev/null 2>&1
        """
    ).strip()


def registry_auth_diagnostic_script(registry: str) -> str:
    """Collect PATH, gcloud, and docker pull errors for setup failures."""
    probe = registry_probe_ref(registry)
    host = registry_host(registry)
    return textwrap.dedent(
        f"""
        set -u
        {docker_auth_preamble(registry)}
        echo "PATH=$PATH"
        echo "gcloud=$(command -v gcloud 2>/dev/null || echo missing)"
        echo "probe={probe}"
        echo "host={host}"
        if ! command -v gcloud >/dev/null 2>&1; then
          echo "gcloud not found after PATH setup"
          exit 0
        fi
        if ! gcloud auth print-access-token >/dev/null 2>&1; then
          echo "gcloud auth print-access-token failed — run: {GCLOUD_AUTH_LOGIN_CMD}"
        fi
        echo "--- docker pull ---"
        sg docker -c "docker pull \\"{probe}\\"" 2>&1 | tail -15
        """
    ).strip()


def ensure_image_registry_auth(
    remote: RemoteHost,
    config: LabConfig,
    *,
    stream: bool = False,
) -> str:
    """Verify Keska mirror access; run interactive gcloud login on lab if needed."""
    registry = config.image_registry
    if not registry:
        return "mirror disabled"
    probe = registry_probe_ref(registry)
    host = registry_host(registry)

    ping = remote.ping()
    if not ping.ok:
        raise RuntimeError(
            f"cannot reach {remote.ssh_target}: {remote.format_failure(ping) or 'ssh failed'}"
        )

    if remote.sh_login(check_registry_auth_script(registry), timeout=180, stream=stream).ok:
        return f"mirror pull ok ({probe})"

    if remote.sh_login(check_registry_auth_ctr_script(registry), timeout=180, stream=stream).ok:
        return f"mirror ctr pull ok ({probe})"

    diag = remote.sh_login(registry_auth_diagnostic_script(registry), timeout=120, stream=stream)
    diag_text = remote.format_failure(diag) if diag.stdout or diag.stderr else diag.stdout.strip()
    has_gcloud = remote.which("gcloud")

    if not has_gcloud:
        raise RuntimeError(
            f"cannot pull {probe}: gcloud not on PATH over SSH (login shell).\n{diag_text}"
        )

    retry = remote.sh_login(configure_gcp_docker_script(registry), timeout=120, stream=stream)
    if retry.ok and remote.sh_login(check_registry_auth_script(registry), timeout=180, stream=stream).ok:
        return f"mirror pull ok ({probe})"
    if retry.ok and remote.sh_login(check_registry_auth_ctr_script(registry), timeout=180, stream=stream).ok:
        return f"mirror ctr pull ok ({probe})"

    needs_login = any(
        token in diag_text.lower()
        for token in (
            "print-access-token failed",
            "denied",
            "403",
            "unauthorized",
            "cannot authenticate",
        )
    )
    if needs_login:
        print(
            f"\nKeska mirror auth required on {remote.ssh_target} ({host}).\n"
            f"Running: {GCLOUD_AUTH_LOGIN_CMD}\n"
            "Open the URL in your browser, then paste the verification code at the SSH prompt.\n",
            flush=True,
        )
        login = remote.run_tty(
            f"bash -lc {shlex.quote(gcloud_auth_login_script())}",
            timeout=None,
        )
        if not login.ok:
            raise RuntimeError(remote.format_failure(login) or f"{GCLOUD_AUTH_LOGIN_CMD} failed")

        cfg = remote.sh_login(configure_gcp_docker_script(registry), timeout=120, stream=stream)
        if not cfg.ok:
            raise RuntimeError(remote.format_failure(cfg) or "gcloud auth configure-docker failed")

        if remote.sh_login(check_registry_auth_script(registry), timeout=180, stream=stream).ok:
            return f"mirror auth ok ({probe})"
        if remote.sh_login(check_registry_auth_ctr_script(registry), timeout=180, stream=stream).ok:
            return f"mirror ctr auth ok ({probe})"

    if "docker daemon" in diag_text.lower() or "docker.sock" in diag_text.lower():
        start = remote.sh_login(ensure_docker_daemon_script(), timeout=90, stream=stream)
        if start.ok and remote.sh_login(check_registry_auth_script(registry), timeout=180, stream=stream).ok:
            return f"mirror pull ok ({probe})"

    raise RuntimeError(f"cannot pull {probe} from Keska mirror.\n{diag_text}")


class ImageRegistryAuthStep(SetupStep):
    """Setup step: verify GCP Artifact Registry credentials on the lab host."""

    name = "image-registry-auth"

    def __init__(self, config: LabConfig):
        self.config = config

    def run(self, remote: "RemoteHost", *, stream: bool = False) -> StepResult:
        try:
            msg = ensure_image_registry_auth(remote, self.config, stream=stream)
            return StepResult(self.name, True, msg)
        except Exception as e:
            return StepResult(self.name, False, str(e))


def ctr_import_base_name(image: str) -> str:
    slug = image.split("/")[-1].split(":")[0]
    return f"docker.io/library/{slug}"


def ctr_pull_with_mirror_script(
    image: str,
    registry: str | None,
    *,
    snapshotter: str | None = None,
) -> str:
    """Shell: ctr pull mirror then upstream; docker import fallback; tag as canonical ref."""
    canonical = ctr_image_ref(image)
    refs = image_pull_refs(image, registry)
    refs_shell = " ".join(shlex.quote(r) for r in refs)
    snap_flag = f" --snapshotter {shlex.quote(snapshotter)}" if snapshotter else ""
    # --local breaks devmapper: pull succeeds but ctr run reports "snapshot does not exist"
    local_flag = " --local" if snapshotter and snapshotter != "devmapper" else ""
    short = image.strip().split("@")[0]
    import_base = shlex.quote(ctr_import_base_name(image))
    snap_name = shlex.quote(snapshotter or "")
    ctr_auth = ctr_mirror_pull_auth_shell(registry)
    return textwrap.dedent(
        f"""
        set -euo pipefail
        {ctr_auth}
        canonical={shlex.quote(canonical)}
        refs=({refs_shell})
        short={shlex.quote(short)}
        pulled=0
        dm_src=""
        for src in "${{refs[@]}}"; do
          if ctr_pull_ref "$src"{snap_flag}{local_flag} --platform linux/amd64 2>/dev/null; then
            if [ "$src" != "$canonical" ]; then
              sudo -n ctr images rm "$canonical" >/dev/null 2>&1 || true
              sudo -n ctr images tag "$src" "$canonical" 2>/dev/null || true
            fi
            pulled=1
            break
          fi
        done
        if [ "$pulled" = 0 ]; then
          {docker_auth_preamble(registry) if registry else lab_path_setup_script()}
          for src in "${{refs[@]}}"; do
            if sg docker -c "docker pull \\"$src\\""; then
              dm_src="$src"
              if [ "$src" != "$short" ]; then
                sg docker -c "docker tag \\"$src\\" \\"$short\\"" >/dev/null 2>&1 || true
              fi
              sg docker -c "docker save \\"$short\\"" | sudo -n ctr images import --base-name {import_base} --digests - >/dev/null
              sudo -n ctr images tag "$short" "$canonical" 2>/dev/null || true
              sudo -n ctr images tag {import_base}:latest "$canonical" 2>/dev/null || true
              if [ -n {snap_name} ]; then
                unpack=${{dm_src:-$canonical}}
                ctr_pull_ref "$unpack"{snap_flag}{local_flag} --platform linux/amd64
                if [ "$unpack" != "$canonical" ]; then
                  sudo -n ctr images tag "$unpack" "$canonical" 2>/dev/null || true
                fi
              fi
              pulled=1
              break
            fi
          done
        fi
        [ "$pulled" = 1 ]
        """
    ).strip()


def docker_pull_with_mirror_script(image: str, registry: str | None) -> str:
    """Shell: docker pull mirror then upstream; tag as short workload name."""
    short = image.strip()
    refs: list[str] = []
    if registry:
        refs.append(mirror_image_ref(short, registry))
    refs.append(short)
    uniq: list[str] = []
    for r in refs:
        if r not in uniq:
            uniq.append(r)
    refs_shell = " ".join(shlex.quote(r) for r in uniq)
    preamble = docker_auth_preamble(registry) if registry else lab_path_setup_script()
    return textwrap.dedent(
        f"""
        set -euo pipefail
        {preamble}
        target={shlex.quote(short)}
        refs=({refs_shell})
        pulled=0
        for src in "${{refs[@]}}"; do
          if sg docker -c "docker pull \\"$src\\""; then
            if [ "$src" != "$target" ]; then
              sg docker -c "docker tag \\"$src\\" \\"$target\\"" >/dev/null 2>&1 || true
            fi
            pulled=1
            break
          fi
        done
        [ "$pulled" = 1 ]
        """
    ).strip()


def all_purge_refs(image: str, registry: str | None) -> list[str]:
    """Image refs to purge from containerd (mirror + canonical + short)."""
    refs = list(image_pull_refs(image, registry))
    refs.append(ctr_image_ref(image))
    short = image.split("@")[0]
    if short not in refs:
        refs.append(short)
    return list(dict.fromkeys(refs))
