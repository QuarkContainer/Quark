"""Small text helpers for logs and error messages."""

from __future__ import annotations

import re


def tail_lines(text: str, n: int = 40) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def format_cmd_output(stdout: str = "", stderr: str = "", *, max_lines: int = 40) -> str:
    """Prefer error-like lines; otherwise return the last N lines of combined output."""
    combined = "\n".join(filter(None, [stdout, stderr]))
    if not combined.strip():
        return "command failed (no output)"

    errish = [
        ln
        for ln in combined.splitlines()
        if re.search(r"\berror\b|\bfailed\b|\bfatal\b|panic!|cannot |can't |-lcap|unable to find library", ln, re.I)
    ]
    if errish:
        text = "\n".join(errish[-max_lines:])
        if "-lcap" in text or "libcap" in text.lower():
            text += "\n\n→ install on lab: sudo apt-get install -y libcap-dev (or re-run lab.quark.run() — installs deps automatically)"
        if "cargo.lock" in text.lower() and "rustlib" in text.lower():
            text += (
                "\n\n→ Quark qkernel needs rust-src on the pinned toolchain:\n"
                "  rustup component add rust-src --toolchain nightly-2024-07-01-x86_64-unknown-linux-gnu"
            )
        return text
    return tail_lines(combined, max_lines)
