#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/vm-exec.sh" bash -lc '
  echo "runc:"
  date +%s%N; docker run --rm ubuntu:24.04 /bin/date +%s%N
  echo "quark_d:"
  date +%s%N; docker run --runtime=quark_d --rm ubuntu:24.04 /bin/date +%s%N
'
