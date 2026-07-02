"""L3 — API spot checks: stats, inspectp, ps."""

from __future__ import annotations

import textwrap


def cri_api_spot_script(
    *,
    runtime_handler: str = "",
    pod_name: str = "keska-cri-l3",
    memory_limit_bytes: int = 67108864,
) -> str:
    rt = f" --runtime={runtime_handler}" if runtime_handler else ""
    rt_label = runtime_handler or "default"
    require_stats_memory = "1" if runtime_handler == "kata" else "0"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        LOG=/tmp/keska-cri-l3-logs
        mkdir -p "$LOG"
        WD=/tmp/keska-cri-l3-$$
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
        cat >"$WD/container.json" <<JSON
{{"metadata":{{"name":"keska-cri-l3-c","namespace":"default"}},"image":{{"image":"docker.io/library/busybox:latest"}},"command":["/bin/sleep","600"],"log_path":"l3.log","linux":{{"resources":{{"memory_limit_in_bytes":{memory_limit_bytes}}}}}}}
JSON

        POD=$(sudo crictl runp{rt} "$WD/pod.json")
        CID=$(sudo crictl create --no-pull "$POD" "$WD/container.json" "$WD/pod.json")
        sudo crictl start "$CID"
        sleep 3

        echo "L3: inspectp"
        sudo crictl inspectp "$POD" | grep -q '"id"' || {{ echo "inspectp failed" >&2; exit 1; }}

        echo "L3: ps"
        sudo crictl ps -q --pod "$POD" | grep -q . || {{ echo "ps empty for pod" >&2; exit 1; }}

        echo "L3: stats"
        sudo crictl stats -o json "$CID" | python3 -c "
import json, sys
require_mem = {require_stats_memory}
data = json.load(sys.stdin)
stats = data.get('stats') or []
if not stats:
    raise SystemExit('stats empty')
row = stats[0]
cid = row.get('attributes', {{}}).get('id', '')
if not cid:
    raise SystemExit('stats missing container id')
print('L3 PASS stats_rpc_ok id=' + cid[:12])
if require_mem:
    mem = row.get('memory') or {{}}
    usage = mem.get('usageBytes') or mem.get('workingSetBytes') or {{}}
    val = int(usage.get('value', '0') or 0)
    if val <= 0:
        raise SystemExit('stats missing memory usage for kata')
    print('L3 PASS stats_mem=' + str(val))
"
        CG=$(sudo crictl inspect "$CID" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('info',{{}}).get('runtimeSpec',{{}}).get('linux',{{}}).get('cgroupsPath',''))")
        MEM_FILE="/sys/fs/cgroup${{CG}}/memory.current"
        if [ -f "$MEM_FILE" ]; then
          MEM=$(cat "$MEM_FILE")
          [ "${{MEM:-0}}" -gt 0 ] || {{ echo "cgroup mem is 0 at $MEM_FILE" >&2; exit 1; }}
          echo "L3 PASS cgroup_mem=$MEM"
        else
          echo "L3 PASS cgroup_host_optional (no $MEM_FILE — stats RPC is gate)"
        fi

        sudo crictl stop -t 30 "$CID"
        sudo crictl rm "$CID"
        CID=""
        sudo crictl stopp "$POD" && sudo crictl rmp "$POD"
        POD=""
        echo "L3 PASS api_spot runtime={rt_label}"
        """
    ).strip()
