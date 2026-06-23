#!/usr/bin/env bash
# Shared helpers for Quark dev scripts.
set -euo pipefail

dev_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

repo_root() {
  cd "$(dev_root)/../.." && pwd
}

require_confirm() {
  local action="$1"
  if [[ "${CONFIRM:-0}" != "1" ]]; then
    echo "error: $action requires CONFIRM=1" >&2
    exit 1
  fi
}
