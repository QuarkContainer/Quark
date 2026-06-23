#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST="${1:-fork}"
RUNTIME="${2:-quark_d}"
DOCKER_FLAGS="${3:---rm}"

exec "${SCRIPT_DIR}/vm-exec.sh" bash -lc "
  set -e
  test -f test/c/${TEST}.c
  gcc -static -o /tmp/qtest test/c/${TEST}.c
  docker run --runtime=${RUNTIME} ${DOCKER_FLAGS} \
    -v /tmp/qtest:/qtest:ro ubuntu:24.04 /qtest
"
