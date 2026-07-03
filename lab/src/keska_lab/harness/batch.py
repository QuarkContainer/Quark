"""Helpers for remote batch benchmark scripts."""

from __future__ import annotations

import textwrap


def remote_batch_loop(n: int, body: str, *, indent: str = "  ") -> str:
    """Wrap a per-iteration bash body in a loop that continues after failures."""
    wrapped = f"( set -euo pipefail\n{body.strip()}\n)"
    indented = textwrap.indent(wrapped, indent)
    return textwrap.dedent(
        f"""
        set +e
        for __batch_i in $(seq 1 {n}); do
        {indented}
        done
        set -e
        """
    ).strip()


def remote_batch_script(*, preamble: str, n: int, body: str) -> str:
    """One SSH script: shared preamble once, then ``n`` timed iterations."""
    lines = ["set -euo pipefail"]
    if preamble.strip():
        lines.append(preamble.strip())
    lines.append(remote_batch_loop(n, body))
    return "\n".join(lines)
