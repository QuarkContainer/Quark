#!/usr/bin/env bash
# Run a command inside the Lima Quark VM with standard dev environment.
set -euo pipefail

LIMA_INSTANCE="${LIMA_INSTANCE:-quark}"
VM_BUILD_DIR="${VM_BUILD_DIR:-~/quark}"
RUSTUP_HOME="${RUSTUP_HOME:-/usr/local/rustup}"
CARGO_HOME="${CARGO_HOME:-/usr/local/cargo}"
AUTO_START="${AUTO_START:-1}"
VERBOSE="${VERBOSE:-0}"

if [[ $# -eq 0 ]]; then
  echo "usage: vm-exec.sh <command...>" >&2
  exit 2
fi

status="$(limactl list "$LIMA_INSTANCE" 2>/dev/null | awk 'NR==2 {print $2}' || true)"
if [[ "$status" != "Running" ]]; then
  if [[ "$AUTO_START" == "1" ]]; then
    echo "→ Lima instance '$LIMA_INSTANCE' is not running; starting..."
    limactl start "$LIMA_INSTANCE"
  else
    echo "error: Lima instance '$LIMA_INSTANCE' is not running (set AUTO_START=1 to auto-start)" >&2
    exit 1
  fi
fi

# Escape single quotes for bash -lc
cmd=$(
  printf '%q ' "$@"
  echo
)

limactl shell "$LIMA_INSTANCE" -- bash -lc "
  set -euo pipefail
  export RUSTUP_HOME='${RUSTUP_HOME}'
  export CARGO_HOME='${CARGO_HOME}'
  export PATH=/usr/local/cargo/bin:\$PATH
  cd ${VM_BUILD_DIR}
  ${cmd}
"
