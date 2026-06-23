#!/usr/bin/env bash
# Preflight checks for Quark dev environment (host + VM).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib.sh"

LIMA_INSTANCE="${LIMA_INSTANCE:-quark}"
FAIL=0

section() { echo; echo "== $1 =="; }

vm_ok() {
  "${SCRIPT_DIR}/vm-exec.sh" "$@"
}

section "Host"
if ! command -v limactl >/dev/null 2>&1; then
  echo "✗ limactl not found (brew install lima)" >&2
  FAIL=1
else
  echo "✓ limactl $(limactl --version 2>/dev/null | head -1 || true)"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "✗ python3 not found" >&2
  FAIL=1
else
  echo "✓ python3 $(python3 --version 2>/dev/null || true)"
fi

if [[ ! -f "$(repo_root)/quark.py" ]]; then
  echo "✗ quark.py not found in repo root" >&2
  FAIL=1
else
  echo "✓ quark.py"
fi

status="$(limactl list "$LIMA_INSTANCE" 2>/dev/null | awk 'NR==2 {print $2}' || true)"
if [[ "$status" == "Running" ]]; then
  echo "✓ Lima '$LIMA_INSTANCE' is Running"
else
  echo "✗ Lima '$LIMA_INSTANCE' status: ${status:-not found}" >&2
  FAIL=1
fi

if [[ "$status" != "Running" ]]; then
  exit "$FAIL"
fi

section "VM / KVM"
if vm_ok kvm-ok >/dev/null 2>&1; then
  echo "✓ /dev/kvm available"
else
  echo "✗ kvm-ok failed" >&2
  FAIL=1
fi

section "VM / Docker"
if vm_ok docker info >/dev/null 2>&1; then
  echo "✓ docker accessible"
  vm_ok docker info 2>/dev/null | grep -E "^ Runtimes:" || true
else
  echo "✗ docker not accessible (try: python3 quark.py vm restart)" >&2
  FAIL=1
fi

section "VM / Quark binaries"
bin_fail=0
for bin in /usr/local/bin/quark_d /usr/local/bin/qkernel_d.bin /usr/local/bin/vdso.so; do
  if vm_ok test -e "$bin"; then
    echo "✓ $bin"
  else
    echo "✗ missing $bin (run: python3 quark.py rebuild)" >&2
    bin_fail=1
  fi
done
if [[ "$bin_fail" -ne 0 ]]; then
  FAIL=1
fi

section "VM / Build tree"
if vm_ok test -f makefile; then
  echo "✓ ~/quark build tree present"
else
  echo "⚠ ~/quark missing (run: python3 quark.py sync)" >&2
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo; echo "Preflight failed." >&2
  exit 1
fi

echo; echo "✓ All preflight checks passed."
