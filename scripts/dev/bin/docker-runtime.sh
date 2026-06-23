#!/usr/bin/env bash
# Register Quark runtimes in Docker daemon.json inside the VM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib.sh"

DAEMON_JSON="${DOCKER_DAEMON_JSON:-/etc/docker/daemon.json}"
VM_REPO_MOUNT="${VM_REPO_MOUNT:-$(repo_root)}"
DAEMON_TEMPLATE="${VM_REPO_MOUNT}/doc/daemon.json"

"${SCRIPT_DIR}/vm-exec.sh" sudo python3 - "$DAEMON_JSON" "$DAEMON_TEMPLATE" <<'PY'
import json
import sys
from pathlib import Path

dest = Path(sys.argv[1])
template = Path(sys.argv[2])
base = {}
if dest.exists() and dest.read_text().strip():
    base = json.loads(dest.read_text())
doc = json.loads(template.read_text())
base.setdefault("runtimes", {})
for name, cfg in doc.get("runtimes", {}).items():
    if name in ("quark", "quark_d"):
        base["runtimes"][name] = cfg
dest.write_text(json.dumps(base, indent=2) + "\n")
print(f"✓ updated {dest} with quark runtimes")
PY

"${SCRIPT_DIR}/vm-exec.sh" sudo systemctl restart docker
echo "✓ docker restarted"
