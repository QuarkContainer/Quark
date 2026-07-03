"""Verify Keska mirror auth on the lab host."""

from __future__ import annotations

import sys

from keska_lab.config import LabConfig
from keska_lab.remote import RemoteHost
from keska_lab.setup.image_registry import ensure_image_registry_auth


def main() -> int:
    cfg = LabConfig.from_env()
    remote = RemoteHost(cfg)
    try:
        msg = ensure_image_registry_auth(remote, cfg)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"OK: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
