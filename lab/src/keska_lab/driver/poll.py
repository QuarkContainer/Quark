from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class WaitStalled(TimeoutError):
    def __init__(self, *, wait_name: str, elapsed_s: float, last_state: str):
        super().__init__(f"wait stalled: {wait_name} elapsed_s={elapsed_s:.2f} last_state={last_state}")
        self.wait_name = wait_name
        self.elapsed_s = elapsed_s
        self.last_state = last_state


class WaitTimedOut(TimeoutError):
    def __init__(self, *, wait_name: str, elapsed_s: float, last_state: str):
        super().__init__(f"wait timed out: {wait_name} elapsed_s={elapsed_s:.2f} last_state={last_state}")
        self.wait_name = wait_name
        self.elapsed_s = elapsed_s
        self.last_state = last_state


@dataclass(frozen=True)
class PollWait(Generic[T]):
    wait_name: str
    check: Callable[[], tuple[bool, T]]
    interval_s: float
    max_wait_s: float
    stall_threshold_s: float
    state_str: Callable[[T], str] = str

    def run(self) -> T:
        t0 = time.monotonic()
        last_change = t0
        last_state: str = ""
        while True:
            ok, state = self.check()
            state_s = self.state_str(state)
            if state_s != last_state:
                last_state = state_s
                last_change = time.monotonic()
            if ok:
                return state

            now = time.monotonic()
            elapsed = now - t0
            if elapsed >= self.max_wait_s:
                raise WaitTimedOut(wait_name=self.wait_name, elapsed_s=elapsed, last_state=last_state)
            if (now - last_change) >= self.stall_threshold_s:
                raise WaitStalled(wait_name=self.wait_name, elapsed_s=elapsed, last_state=last_state)
            time.sleep(self.interval_s)

