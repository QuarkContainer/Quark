from __future__ import annotations

import time
from contextlib import contextmanager


class OpBudgetExceeded(TimeoutError):
    def __init__(self, *, budget_s: float, elapsed_s: float):
        super().__init__(f"op budget exceeded: budget_s={budget_s} elapsed_s={elapsed_s}")
        self.budget_s = budget_s
        self.elapsed_s = elapsed_s


@contextmanager
def op_deadline(*, budget_s: float):
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        if elapsed > budget_s:
            raise OpBudgetExceeded(budget_s=budget_s, elapsed_s=elapsed)

