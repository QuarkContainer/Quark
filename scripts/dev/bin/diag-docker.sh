#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/vm-exec.sh" docker info >/dev/null
"${SCRIPT_DIR}/vm-exec.sh" docker info 2>/dev/null | grep -E "^ Runtimes:" | grep -q quark || {
  echo "✗ quark runtime not registered (run: python3 quark.py docker setup)" >&2
  exit 1
}
echo "✓ docker OK, quark runtime registered"
