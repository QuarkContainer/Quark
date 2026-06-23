#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/vm-exec.sh" bash -lc '
  while true; do
    clear
    date
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" 2>/dev/null || true
    pgrep -a quark 2>/dev/null || true
    sleep 2
  done
'
