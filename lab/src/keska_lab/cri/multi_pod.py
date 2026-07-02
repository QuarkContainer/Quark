"""L5 — multi-container pod (pause + two workloads)."""

from __future__ import annotations

import textwrap


def cri_multi_container_smoke_script(
    *,
    runtime_handler: str = "",
    pod_name: str = "keska-cri-l5",
) -> str:
    """Two busybox containers in one pod — exercises subcontainer paths (bug 039)."""
    rt = f" --runtime={runtime_handler}" if runtime_handler else ""
    return textwrap.dedent(
        f"""
        set -euo pipefail
        LOG=/tmp/keska-cri-l5-logs
        mkdir -p "$LOG"
        WD=/tmp/keska-cri-l5-$$
        mkdir -p "$WD"
        POD=""
        CID1=""
        CID2=""
        wait_stopped() {{
          local cid="$1"
          local i
          for i in $(seq 1 60); do
            if ! sudo crictl ps -q --no-trunc 2>/dev/null | grep -Fx "$cid" >/dev/null; then
              return 0
            fi
            sleep 1
          done
          echo "timeout waiting for container $cid to exit" >&2
          return 1
        }}

        stop_rm() {{
          local cid="$1"
          [ -n "$cid" ] || return 0
          sudo crictl stop -t 60 "$cid"
          wait_stopped "$cid"
          sudo crictl rm "$cid"
        }}

        cleanup() {{
          stop_rm "$CID1" 2>/dev/null || true
          stop_rm "$CID2" 2>/dev/null || true
          CID1="" CID2=""
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
        cat >"$WD/c1.json" <<'JSON'
{{"metadata":{{"name":"keska-l5-a","namespace":"default"}},"image":{{"image":"docker.io/library/busybox:latest"}},"command":["/bin/sleep","600"],"log_path":"a.log"}}
JSON
        cat >"$WD/c2.json" <<'JSON'
{{"metadata":{{"name":"keska-l5-b","namespace":"default"}},"image":{{"image":"docker.io/library/busybox:latest"}},"command":["/bin/sleep","600"],"log_path":"b.log"}}
JSON

        POD=$(sudo crictl runp{rt} "$WD/pod.json")
        CID1=$(sudo crictl create --no-pull "$POD" "$WD/c1.json" "$WD/pod.json")
        sudo crictl start "$CID1"
        CID2=$(sudo crictl create --no-pull "$POD" "$WD/c2.json" "$WD/pod.json")
        sudo crictl start "$CID2"

        COUNT=$(sudo crictl ps -q --pod "$POD" | wc -l | tr -d ' ')
        [ "$COUNT" -ge 2 ] || {{ echo "expected 2 containers got $COUNT" >&2; exit 1; }}

        sudo crictl exec "$CID1" /bin/sh -c 'echo pod-a' | grep -q pod-a
        sudo crictl exec "$CID2" /bin/sh -c 'echo pod-b' | grep -q pod-b

        stop_rm "$CID1"
        stop_rm "$CID2"
        CID1="" CID2=""
        sudo crictl stopp "$POD" && sudo crictl rmp "$POD"
        POD=""
        echo "L5 PASS multi_container runtime={runtime_handler or 'default'}"
        """
    ).strip()
