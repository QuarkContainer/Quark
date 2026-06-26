"""Centralized lab sandbox cleanup (Quark + Kata) on the lab host."""

from __future__ import annotations

from keska_lab.remote import RemoteHost

CLEANUP_SCRIPT = """
set -euo pipefail
sudo -n killall -9 firecracker 2>/dev/null || true
pkill -9 -f '[q]uark boot' 2>/dev/null || true
pkill -9 -f '[q]uark_d boot' 2>/dev/null || true
pkill -9 qemu 2>/dev/null || true
for ns in default k8s.io; do
  ids=$(sudo -n ctr --namespace "$ns" containers ls -q 2>/dev/null | grep -E '^keska-' || true)
  for id in $ids; do
    timeout 3 sudo -n ctr --namespace "$ns" task kill -s SIGKILL "$id" 2>/dev/null || true
    timeout 3 sudo -n ctr --namespace "$ns" containers rm "$id" 2>/dev/null || true
  done
done
for bin in quark quark_d; do
  command -v "$bin" >/dev/null 2>&1 || continue
  ids=$(timeout 5 "$bin" list 2>/dev/null | awk 'NR>1 {print $1}' || true)
  for id in $ids; do
    timeout 10 sudo -n "$bin" delete --force "$id" 2>/dev/null || true
  done
done
sudo rm -rf /tmp/keska-lab/pg-run-* 2>/dev/null || true
sg docker -c 'docker ps -aq --filter name=keska- 2>/dev/null | xargs -r docker rm -f' >/dev/null 2>&1 || true
""".strip()


def cleanup_quark_sandboxes(remote: RemoteHost, *, timeout: int = 120) -> None:
    remote.sh(CLEANUP_SCRIPT, timeout=timeout)
