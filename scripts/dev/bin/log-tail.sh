#!/usr/bin/env bash
# Tail, cat, or clear Quark logs inside the VM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QLOG_DIR="${QLOG_DIR:-/var/log/quark}"
QLOG_FILE="${QLOG_FILE:-/var/log/quark/quark.log}"
ACTION="${1:-tail}"
LINES="${LINES:-200}"

case "$ACTION" in
  tail)
    "${SCRIPT_DIR}/vm-exec.sh" bash -lc "sudo mkdir -p ${QLOG_DIR}; sudo touch ${QLOG_FILE}; sudo tail -F ${QLOG_FILE}"
    ;;
  cat)
    "${SCRIPT_DIR}/vm-exec.sh" bash -lc "sudo tail -n ${LINES} ${QLOG_FILE} 2>/dev/null || echo '(no log yet)'"
    ;;
  clear)
    if [[ "${CONFIRM:-0}" != "1" ]]; then
      echo "error: logs:clear requires CONFIRM=1" >&2
      exit 1
    fi
    "${SCRIPT_DIR}/vm-exec.sh" sudo bash -lc "rm -f ${QLOG_DIR}/*"
    echo "✓ cleared ${QLOG_DIR}"
    ;;
  *)
    echo "usage: log-tail.sh <tail|cat|clear>" >&2
    exit 2
    ;;
esac
