"""Bridge CNI gate — CRI lifecycle smoke."""

from __future__ import annotations

from keska_lab.cri.lifecycle import cri_lifecycle_smoke_script


def bridge_gate_l1_script(*, runtime_handler: str = "quark") -> str:
    return cri_lifecycle_smoke_script(runtime_handler=runtime_handler)
