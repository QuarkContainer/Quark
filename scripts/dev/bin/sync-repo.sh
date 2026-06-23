#!/usr/bin/env bash
# Sync repo from virtiofs mount to fast VM-local build directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIMA_INSTANCE="${LIMA_INSTANCE:-quark}"
VM_REPO_MOUNT="${VM_REPO_MOUNT:-}"
VM_BUILD_DIR="${VM_BUILD_DIR:-~/quark}"
SYNC_EXCLUDES="${SYNC_EXCLUDES:-target,build,.git}"
AUTO_START="${AUTO_START:-1}"

if [[ -z "$VM_REPO_MOUNT" ]]; then
  echo "error: VM_REPO_MOUNT is not set" >&2
  exit 1
fi

status="$(limactl list "$LIMA_INSTANCE" 2>/dev/null | awk 'NR==2 {print $2}' || true)"
if [[ "$status" != "Running" ]]; then
  if [[ "$AUTO_START" == "1" ]]; then
    echo "→ Lima instance '$LIMA_INSTANCE' is not running; starting..."
    limactl start "$LIMA_INSTANCE"
  else
    echo "error: Lima instance '$LIMA_INSTANCE' is not running" >&2
    exit 1
  fi
fi

exclude_flags=""
IFS=',' read -r -a excludes <<< "$SYNC_EXCLUDES"
for item in "${excludes[@]}"; do
  item="${item// /}"
  [[ -n "$item" ]] || continue
  exclude_flags+=" --exclude=${item}"
done

limactl shell "$LIMA_INSTANCE" -- bash -lc "
  set -euo pipefail
  mkdir -p ${VM_BUILD_DIR}
  rsync -a --delete${exclude_flags} ${VM_REPO_MOUNT}/ ${VM_BUILD_DIR}/
  echo '✓ synced ${VM_REPO_MOUNT} → ${VM_BUILD_DIR}'
"
