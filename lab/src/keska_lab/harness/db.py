"""Database benchmark helpers (pgbench parsing)."""

from __future__ import annotations

import re


def parse_pgbench_tps(output: str) -> float:
    for line in output.splitlines():
        m = re.search(r"tps\s*=\s*([\d.]+)", line, re.I)
        if m:
            return float(m.group(1))
    raise RuntimeError("no TPS in pgbench output")
