"""Plain functions for common lab workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from keska_lab.setup.base import SetupReport

if TYPE_CHECKING:
    from keska_lab.session import LabSession


def build_quark(lab: LabSession, *, stream: bool = True) -> SetupReport:
    """Build and install Quark — same as ``lab.quark.run()``."""
    return lab.quark.run(stream=stream)
