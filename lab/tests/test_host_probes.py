from __future__ import annotations

from keska_lab.host.probes import OrphanReport, ProcMatch, terminate_processes


def test_orphan_report_empty():
    r = OrphanReport()
    assert r.is_empty()


def test_terminate_processes_noop_for_missing_pids():
    terminate_processes([ProcMatch(pid=999999, name="missing")], timeout_s=0.01)

