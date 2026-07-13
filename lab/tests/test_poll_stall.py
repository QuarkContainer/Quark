from __future__ import annotations

import time

import pytest

from keska_lab.driver.poll import PollWait, WaitStalled, WaitTimedOut


def test_pollwait_stalls_fast(monkeypatch):
    # Avoid real sleeps.
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    state = {"x": 0}

    def check():
        # Never progresses, always false.
        return False, state["x"]

    w = PollWait(
        wait_name="test",
        check=check,
        interval_s=0.01,
        max_wait_s=10.0,
        stall_threshold_s=0.05,
        state_str=str,
    )
    with pytest.raises(WaitStalled):
        w.run()


def test_pollwait_times_out_if_changes_but_never_ok(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    state = {"x": 0}
    t0 = time.monotonic()

    def check():
        # Keep changing so stall doesn't trigger.
        state["x"] += 1
        return False, state["x"]

    w = PollWait(
        wait_name="test_timeout",
        check=check,
        interval_s=0.0,
        max_wait_s=0.02,
        stall_threshold_s=10.0,
        state_str=str,
    )
    with pytest.raises(WaitTimedOut):
        w.run()
    assert time.monotonic() - t0 >= 0

