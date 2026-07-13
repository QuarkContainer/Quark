from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from keska_lab.driver.errors import DriverErrorCode, DriverStatus


class CleanReport(BaseModel):
    pods_remaining: int = 0
    containers_remaining: int = 0
    orphan_shims: int = 0
    orphan_firecrackers: int = 0


class DriverError(BaseModel):
    code: DriverErrorCode
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class DriverEnvelope(BaseModel):
    protocol_version: Literal[1] = 1
    request_id: str
    op: str
    status: DriverStatus
    duration_s: float = 0.0
    error: DriverError | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    clean_report: CleanReport = Field(default_factory=CleanReport)

    def to_json(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())


@dataclass(frozen=True)
class ExitCode:
    ok: int = 0
    op_failed: int = 1
    platform_blocked: int = 2
    protocol_mismatch: int = 10
    input_invalid: int = 11
    host_not_ready: int = 12
    teardown_leak: int = 13
    internal: int = 14


def exit_code_for_envelope(env: DriverEnvelope) -> int:
    if env.status == DriverStatus.ok:
        if env.clean_report.pods_remaining or env.clean_report.containers_remaining:
            return ExitCode.teardown_leak
        if env.clean_report.orphan_shims or env.clean_report.orphan_firecrackers:
            return ExitCode.teardown_leak
        return ExitCode.ok
    if env.status == DriverStatus.platform_blocked:
        return ExitCode.platform_blocked
    return ExitCode.op_failed

