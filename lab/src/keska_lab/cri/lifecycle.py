"""L2 — full pod lifecycle: runp → run → exec → teardown."""

from __future__ import annotations

import textwrap


def cri_lifecycle_smoke_script(
    *,
    runtime_handler: str = "",
    pod_name: str = "keska-cri-l2",
) -> str:
    """Shell script: pod sandbox + workload + exec + clean teardown.

    Exercises bugs 033–039 chain (VM create, cgroup, boot, pivot, subcontainer).
    """
    rt = f" --runtime={runtime_handler}" if runtime_handler else ""
    return textwrap.dedent(
        f"""
        set -euo pipefail
        LOG=/tmp/keska-cri-l2-logs
        mkdir -p "$LOG"
        WD=/tmp/keska-cri-l2-$$
        mkdir -p "$WD"
        POD=""
        CID=""
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
{{"metadata":{{"name":"{pod_name}-$$","uid":"{pod_name}-uid-$$","namespace":"default"}},"log_directory":"$LOG","linux":{{}}}}
JSON
        cat >"$WD/container.json" <<'JSON'
{{"metadata":{{"name":"keska-cri-l2-c","namespace":"default"}},"image":{{"image":"docker.io/library/busybox:latest"}},"command":["/bin/sleep","600"],"log_path":"l2.log"}}
JSON

        echo "L2: runp (pod sandbox)"
        POD=$(sudo crictl runp{rt} "$WD/pod.json")
        [ -n "$POD" ] || {{ echo "runp returned empty pod id" >&2; exit 1; }}

        echo "L2: run workload container"
        CID=$(sudo crictl create --no-pull "$POD" "$WD/container.json" "$WD/pod.json")
        sudo crictl start "$CID"
        [ -n "$CID" ] || {{ echo "run returned empty cid" >&2; exit 1; }}

        echo "L2: exec"
        OUT=$(sudo crictl exec "$CID" /bin/sh -c 'echo hello-l2')
        [ "$OUT" = "hello-l2" ] || {{ echo "exec expected hello-l2 got: $OUT" >&2; exit 1; }}

        echo "L2: stop container"
        sudo crictl stop -t 30 "$CID"
        sudo crictl rm "$CID"
        CID=""

        echo "L2: stop and remove pod"
        sudo crictl stopp "$POD"
        sudo crictl rmp "$POD"
        POD=""

        echo "L2 PASS pod_lifecycle ok runtime={runtime_handler or 'default'}"
        """
    ).strip()
