#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_REPO_MOUNT="${VM_REPO_MOUNT:?VM_REPO_MOUNT not set}"

"${SCRIPT_DIR}/vm-exec.sh" test -f "${VM_REPO_MOUNT}/makefile" || {
  echo "✗ mount missing makefile at ${VM_REPO_MOUNT}" >&2
  exit 1
}
echo "✓ repo mount OK: ${VM_REPO_MOUNT}"

if "${SCRIPT_DIR}/vm-exec.sh" test -f makefile; then
  echo "✓ ~/quark synced"
else
  echo "⚠ ~/quark not synced (run: python3 quark.py sync)"
fi
