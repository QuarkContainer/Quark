"""Unit tests for node gate script contracts."""

from __future__ import annotations

from keska_lab.gate.bridge_gate import bridge_gate_l1_script
from keska_lab.gate.tsot_gate import minimal_pod_def, tsot_gate_l1_script, tsot_register_uid_script


def test_minimal_pod_def_status_is_struct():
    pod = minimal_pod_def("test-uid")
    assert isinstance(pod["status"], dict)
    assert pod["status"]["phase"] == "Pending"


def test_bridge_gate_has_lifecycle():
    s = bridge_gate_l1_script(runtime_handler="quark")
    assert "crictl runp" in s
    assert "trap cleanup EXIT" in s


def test_tsot_gate_l1_has_create_and_lookup():
    s = tsot_gate_l1_script("/home/lab/Quark")
    assert "CreatePod" in s
    assert "GetPodSandboxAddr" in s
    assert "trap cleanup EXIT" in s
    assert "TSOT L1 PASS" in s


def test_tsot_register_uid_uses_grpcurl():
    s = tsot_register_uid_script("/home/lab/Quark", "abc-uid")
    assert "grpcurl" in s
    assert "abc-uid" in s
    assert "NodeAgentService/CreatePod" in s
