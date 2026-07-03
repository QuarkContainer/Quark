"""OOP setup pipeline — run before each benchmark set."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from keska_lab.remote import RemoteHost


from keska_lab.textutil import tail_lines


@dataclass
class StepResult:
    name: str
    ok: bool
    message: str
    duration_s: float = 0.0


@dataclass
class SetupReport:
    pipeline: str
    steps: list[StepResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def total_s(self) -> float:
        return sum(s.duration_s for s in self.steps)

    def raise_if_failed(self) -> SetupReport:
        failed = [s for s in self.steps if not s.ok]
        if failed:
            parts = []
            for s in failed:
                msg = s.message.strip()
                if "\n" in msg:
                    msg = tail_lines(msg, 15)
                parts.append(f"{s.name}: {msg}")
            raise RuntimeError("setup failed — " + "; ".join(parts))
        return self


class SetupStep(ABC):
    """One idempotent preparation step."""

    name: str

    @abstractmethod
    def run(self, remote: RemoteHost) -> StepResult:
        ...


class SetupPipeline:
    """
    Fluent builder for benchmark preparation.

    Example:
        SetupPipeline("quark-bench")
            .add(SyncQuarkSourcesStep(repo))
            .add(BuildQuarkOnLabStep("debug"))
            .run(remote)
    """

    def __init__(self, name: str):
        self.name = name
        self._steps: list[SetupStep] = []

    def add(self, step: SetupStep) -> SetupPipeline:
        self._steps.append(step)
        return self

    def extend(self, steps: list[SetupStep]) -> SetupPipeline:
        self._steps.extend(steps)
        return self

    def run(self, remote: RemoteHost) -> SetupReport:
        report = SetupReport(pipeline=self.name)
        for step in self._steps:
            t0 = time.perf_counter()
            try:
                result = step.run(remote)
            except Exception as e:
                result = StepResult(step.name, False, str(e), time.perf_counter() - t0)
            result.duration_s = time.perf_counter() - t0
            report.steps.append(result)
            if not result.ok:
                break
        return report


class CallableStep(SetupStep):
    """Wrap a function as a setup step."""

    def __init__(self, name: str, fn: Callable[[RemoteHost], str]):
        self.name = name
        self._fn = fn

    def run(self, remote: RemoteHost) -> StepResult:
        try:
            msg = self._fn(remote)
            return StepResult(self.name, True, msg)
        except Exception as e:
            return StepResult(self.name, False, str(e))
