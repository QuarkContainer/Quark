#!/usr/bin/env bash
# Set DebugLevel or ShimMode in /etc/quark/config.json inside the VM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QCONFIG="${QCONFIG:-/etc/quark/config.json}"
ACTION="${1:-}"

if [[ -z "$ACTION" ]]; then
  echo "usage: config-level.sh <debug|quiet|shim-on|shim-off|show>" >&2
  exit 2
fi

"${SCRIPT_DIR}/vm-exec.sh" sudo python3 - "$QCONFIG" "$ACTION" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
action = sys.argv[2]
cfg = json.loads(path.read_text())
if action == "debug":
    cfg["DebugLevel"] = "Debug"
elif action == "quiet":
    cfg["DebugLevel"] = "Error"
elif action == "shim-on":
    cfg["ShimMode"] = True
elif action == "shim-off":
    cfg["ShimMode"] = False
elif action == "show":
    print(json.dumps(cfg, indent=2))
    sys.exit(0)
else:
    raise SystemExit(f"unknown action: {action}")
path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"✓ updated {path} ({action})")
PY
