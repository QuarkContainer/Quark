#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in /usr/local/bin/quark_d /usr/local/bin/qkernel_d.bin /usr/local/bin/vdso.so; do
  "${SCRIPT_DIR}/vm-exec.sh" test -e "$f" || { echo "✗ missing $f"; exit 1; }
  "${SCRIPT_DIR}/vm-exec.sh" file "$f"
done
echo "✓ binaries present"
