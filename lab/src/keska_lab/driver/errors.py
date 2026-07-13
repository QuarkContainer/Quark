from __future__ import annotations

from enum import Enum


class DriverErrorCode(str, Enum):
    # Protocol / input
    protocol_mismatch = "PROTOCOL_MISMATCH"
    input_invalid = "INPUT_INVALID"
    protocol_invalid = "PROTOCOL_INVALID"

    # Timeouts / budgets
    op_budget_exceeded = "OP_BUDGET_EXCEEDED"
    ssh_driver_timeout = "SSH_DRIVER_TIMEOUT"
    cri_command_timeout = "CRI_COMMAND_TIMEOUT"
    wait_stalled = "WAIT_STALLED"
    grpc_deadline_exceeded = "GRPC_DEADLINE_EXCEEDED"

    # Lifecycle / cleanup
    teardown_leak = "TEARDOWN_LEAK"
    host_not_ready = "HOST_NOT_READY"

    # Platform / bench
    platform_blocked = "PLATFORM_BLOCKED"
    iperf_no_throughput = "IPERF_NO_THROUGHPUT"


class DriverStatus(str, Enum):
    ok = "ok"
    failed = "failed"
    platform_blocked = "platform_blocked"

