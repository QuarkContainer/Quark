"""Unit tests for E2 MmapRead coherency harness."""

from __future__ import annotations

import pytest

from keska_lab.coherency.mmap_read import COHERENCY_CASES, _guest_pass_line
from keska_lab.harness.network import python_exec_cmd


def test_coherency_case_ids() -> None:
    ids = [c.id for c in COHERENCY_CASES]
    assert ids == ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def test_guest_pass_line() -> None:
    assert _guest_pass_line("noise\nPASS C2\n", "PASS C2")
    assert not _guest_pass_line("FAIL\n", "PASS C2")


@pytest.mark.parametrize("case_id", ["C2", "C3", "C4", "C5", "C7"])
def test_guest_python_payloads_compile(case_id: str) -> None:
    from keska_lab.coherency import mmap_read as mr

    mapping = {
        "C2": mr.C2_GUEST_PY,
        "C3": mr.C3_GUEST_PY,
        "C4": mr.C4_GUEST_PY,
        "C5": mr.C5_GUEST_PY,
        "C7": mr.C7_GUEST_PY,
    }
    code = mapping[case_id]
    compile(code, f"<{case_id}>", "exec")


def test_python_exec_cmd_roundtrip() -> None:
    code = "print('ok')"
    cmd = python_exec_cmd(code)
    assert "base64" in cmd
