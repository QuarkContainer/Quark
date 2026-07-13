"""Centralized lab sandbox cleanup (Quark + Kata) on the lab host."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from keska_lab.remote import RemoteHost

CLEANUP_SCRIPT_TEMPLATE = """
set -euo pipefail
sudo -n killall -9 firecracker 2>/dev/null || true
sudo -n pkill -9 -f '[q]uark exec' 2>/dev/null || true
sudo -n pkill -9 -f '[q]uark_d exec' 2>/dev/null || true
sudo -n pkill -9 -f '[q]uark boot' 2>/dev/null || true
sudo -n pkill -9 qemu 2>/dev/null || true
{orphan_kill}
if command -v crictl >/dev/null 2>&1 && [ -S /run/containerd/containerd.sock ]; then
  pods=$(sudo -n crictl pods -q --name 'keska' 2>/dev/null || true)
  for pod in $pods; do
    timeout 5 sudo -n crictl stopp "$pod" 2>/dev/null || true
    timeout 5 sudo -n crictl rmp -f "$pod" 2>/dev/null || true
  done
fi
for ns in default k8s.io; do
  ids=$(sudo -n ctr --namespace "$ns" containers ls -q 2>/dev/null | grep -E '^keska-' || true)
  for id in $ids; do
    timeout 3 sudo -n ctr --namespace "$ns" task kill -s SIGKILL "$id" 2>/dev/null || true
    timeout 3 sudo -n ctr --namespace "$ns" containers rm "$id" 2>/dev/null || true
  done
done
for bin in quark quark_d; do
  command -v "$bin" >/dev/null 2>&1 || continue
  ids=$(timeout 5 sudo -n "$bin" list 2>/dev/null | awk 'NR>1 {print $1}' || true)
  for id in $ids; do
    timeout 10 sudo -n "$bin" delete --force "$id" 2>/dev/null || true
  done
done
# Stopped sandboxes often survive delete; stale metadata breaks the next create/start.
sudo rm -rf /var/lib/quark/keska-* /var/lib/quark/keska_* 2>/dev/null || true
sudo rm -rf /run/qvisor/keska-* /run/qvisor/keska_* /run/qvisor/chk* 2>/dev/null || true
sudo mkdir -p /var/log/quark /var/run/quark
sudo chmod 1777 /var/log/quark /var/run/quark 2>/dev/null || true
sudo rm -rf /tmp/keska-lab/pg-run-* /tmp/keska-cri-* /tmp/keska-iperf-* /tmp/keska-kata-pgdata-* 2>/dev/null || true
for mp in /tmp/keska-kata-pgdata-*; do
  [ -d "$mp" ] || continue
  sudo -n umount "$mp" 2>/dev/null || true
done
sudo rm -rf {io_bench_dir}/run-* 2>/dev/null || true
sg docker -c 'docker ps -aq --filter name=keska- 2>/dev/null | xargs -r docker rm -f' >/dev/null 2>&1 || true
# If ghost metadata remains, wipe keska entries outright.
left=$(timeout 5 sudo -n quark list 2>/dev/null | awk 'NR>1 {print $1}' | wc -l || echo 0)
if [ "${left:-0}" -gt 20 ]; then
  sudo rm -rf /run/qvisor/keska-* /run/qvisor/keska_* /run/qvisor/chk* 2>/dev/null || true
fi
""".strip()

ORPHAN_KILL_SCRIPT = """
sudo -n pkill -9 -f '[q]uark create' 2>/dev/null || true
sudo -n pkill -9 -f '[q]uark delete' 2>/dev/null || true
sudo -n pkill -9 -f '[q]uark list' 2>/dev/null || true
sudo -n pkill -9 -f 'timeout.*crictl exec.*iperf' 2>/dev/null || true
sudo -n pkill -9 -f '[c]rictl exec.*iperf3 -c' 2>/dev/null || true
sudo -n pkill -9 -f 'keska-net-test' 2>/dev/null || true
sudo -n pkill -9 -f 'keska-tsot-net' 2>/dev/null || true
""".strip()

ORPHAN_SCAN_SCRIPT = r"""
set -euo pipefail
scan() {
  local label="$1"
  shift
  ps aux 2>/dev/null | grep -E "$@" | grep -v grep || true
}
emit() {
  local label="$1"
  local line="$2"
  local pid
  pid=$(echo "$line" | awk '{print $2}')
  [ -n "$pid" ] || return 0
  echo "${label}|${pid}|$(echo "$line" | awk '{$1=$2=""; sub(/^ /,""); print}')"
}
while IFS= read -r line; do emit quark_create "$line"; done < <(scan quark_create '[q]uark create')
while IFS= read -r line; do emit quark_boot "$line"; done < <(scan quark_boot '[q]uark boot')
while IFS= read -r line; do emit quark_list "$line"; done < <(scan quark_list '[q]uark list')
while IFS= read -r line; do emit crictl_iperf "$line"; done < <(scan crictl_iperf 'timeout.*crictl exec.*iperf|[c]rictl exec.*iperf3 -c')
while IFS= read -r line; do emit keska_net_test "$line"; done < <(scan keska_net_test 'keska-net-test|keska-tsot-net')
""".strip()


@dataclass
class OrphanProcess:
    category: str
    pid: str
    command: str


@dataclass
class OrphanReport:
    processes: list[OrphanProcess] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.processes)

    def is_empty(self) -> bool:
        return self.total == 0

    def by_category(self) -> dict[str, list[OrphanProcess]]:
        out: dict[str, list[OrphanProcess]] = {}
        for proc in self.processes:
            out.setdefault(proc.category, []).append(proc)
        return out

    def warning_lines(self, *, when: str) -> list[str]:
        if self.is_empty():
            return []
        lines = [
            f"lab orphan processes detected ({when}): {self.total} stale process(es) — "
            "likely a prior bench/teardown bug if this repeats"
        ]
        for category, procs in sorted(self.by_category().items()):
            sample = ", ".join(f"pid={p.pid}" for p in procs[:3])
            extra = f" (+{len(procs) - 3} more)" if len(procs) > 3 else ""
            lines.append(f"  {category}: {len(procs)} ({sample}{extra})")
        return lines


def orphan_scan_script() -> str:
    return ORPHAN_SCAN_SCRIPT


def orphan_kill_script() -> str:
    return ORPHAN_KILL_SCRIPT


def parse_orphan_scan(output: str) -> OrphanReport:
    processes: list[OrphanProcess] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        category, pid, command = line.split("|", 2)
        processes.append(OrphanProcess(category=category, pid=pid, command=command))
    return OrphanReport(processes=processes)


def scan_orphans(remote: RemoteHost, *, timeout: int = 20) -> OrphanReport:
    result = remote.sh(orphan_scan_script(), timeout=timeout, check=False)
    if not result.ok:
        return OrphanReport()
    return parse_orphan_scan(result.stdout)


def cleanup_script(*, io_bench_dir: str = "/var/lib/keska-lab/io-bench") -> str:
    root = io_bench_dir.rstrip("/")
    body = CLEANUP_SCRIPT_TEMPLATE.replace("{orphan_kill}", orphan_kill_script())
    return body.replace("{io_bench_dir}", shlex.quote(root))


def cleanup_quark_sandboxes(
    remote: RemoteHost,
    *,
    timeout: int = 120,
    io_bench_dir: str | None = None,
) -> OrphanReport:
    """Cleanup sandboxes and return any orphan processes found beforehand."""
    before = scan_orphans(remote)
    bench_dir = io_bench_dir or remote.config.io_bench_dir
    remote.sh(cleanup_script(io_bench_dir=bench_dir), timeout=timeout)
    return before
