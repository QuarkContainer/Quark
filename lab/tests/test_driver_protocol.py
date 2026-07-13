from __future__ import annotations

import json

import pytest

from keska_lab.driver.errors import DriverErrorCode, DriverStatus
from keska_lab.driver.protocol import DriverEnvelope, exit_code_for_envelope


def test_driver_envelope_roundtrip_and_exit_code_ok():
    env = DriverEnvelope(
        request_id="123",
        op="network.inet_connect",
        status=DriverStatus.ok,
        duration_s=1.2,
        result={"connect_ms": 12.3},
    )
    d = env.to_json()
    assert d["protocol_version"] == 1
    assert d["status"] == "ok"
    assert d["result"]["connect_ms"] == 12.3
    assert exit_code_for_envelope(env) == 0


def test_driver_envelope_teardown_leak_exit_code():
    env = DriverEnvelope(
        request_id="123",
        op="network.inet_connect",
        status=DriverStatus.ok,
        clean_report={"pods_remaining": 1},
    )
    assert exit_code_for_envelope(env) == 13


def test_driver_envelope_platform_blocked_exit_code():
    env = DriverEnvelope(
        request_id="123",
        op="network.iperf_pair",
        status=DriverStatus.platform_blocked,
        error={
            "code": DriverErrorCode.platform_blocked,
            "message": "guest panic",
        },
    )
    assert exit_code_for_envelope(env) == 2


def test_driver_envelope_rejects_unknown_status():
    bad = {
        "protocol_version": 1,
        "request_id": "123",
        "op": "x",
        "status": "maybe",
        "duration_s": 0,
        "result": {},
        "clean_report": {},
    }
    with pytest.raises(Exception):
        DriverEnvelope.model_validate(bad)

