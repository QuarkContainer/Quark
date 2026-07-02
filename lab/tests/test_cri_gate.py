"""Unit tests for CRI gate — harness contracts (first protection layer)."""

from __future__ import annotations

import json

from keska_lab.cri.api_spot import cri_api_spot_script
from keska_lab.cri.bisect import BUG_CHAIN, all_cri_files, check_032_local
from keska_lab.cri.gate import LAYER_SCRIPTS, LAYER_TIMEOUT, runtime_flag
from keska_lab.cri.lifecycle import cri_lifecycle_smoke_script
from keska_lab.cri.multi_pod import cri_multi_container_smoke_script
from keska_lab.setup.quark_config import cri_bench_config_json, deploy_config_script


def test_l2_includes_runp_exec_teardown():
    s = cri_lifecycle_smoke_script(runtime_handler="quark")
    assert "crictl runp" in s
    assert "crictl create" in s
    assert "crictl start" in s
    assert "crictl exec" in s
    assert "hello-l2" in s
    assert "crictl stopp" in s
    assert "crictl rmp" in s
    assert "trap cleanup EXIT" in s
    assert "--runtime=quark" in s
    assert "crictl run " not in s  # legacy path without runp


def test_l2_default_runtime_no_flag():
    s = cri_lifecycle_smoke_script()
    assert "runp " in s
    assert "--runtime=" not in s


def test_l3_stats_and_inspectp():
    s = cri_api_spot_script(runtime_handler="kata")
    assert "crictl inspectp" in s
    assert "crictl stats -o json" in s
    assert "--runtime=kata" in s
    assert "memory_limit_in_bytes" in s
    assert "stats missing memory usage for kata" in s
    assert "cgroup_host_optional" in s


def test_l3_quark_skips_stats_memory_requirement():
    s = cri_api_spot_script()
    assert "require_mem = 0" in s
    assert "cgroup_host_optional" in s


def test_l2_uses_create_start_after_runp():
    s = cri_lifecycle_smoke_script(runtime_handler="quark")
    assert "crictl create" in s
    assert "crictl start" in s
    assert 'create --no-pull "$POD"' in s


def test_l5_create_start_after_runp():
    s = cri_multi_container_smoke_script()
    assert s.count("crictl create") == 2
    assert s.count("crictl start") == 2


def test_l5_sequential_stop_with_wait_stopped():
    s = cri_multi_container_smoke_script()
    assert "wait_stopped" in s
    assert "stop_rm" in s
    assert "stop -t 60" in s
    # sequential teardown — not batch rm -f both at once
    assert s.index("stop_rm") < s.rindex("stop_rm")


def test_containerd_cri_setup_uses_l2_lifecycle():
    from keska_lab.setup.containerd_cri import cri_lifecycle_setup_smoke_script

    s = cri_lifecycle_setup_smoke_script()
    assert "crictl runp" in s
    assert "crictl create" in s
    assert "crictl exec" in s
    assert "hello-l2" in s
    assert "crictl run " not in s


def test_cri_bench_config_profile():
    cfg = cri_bench_config_json()
    assert cfg["Sandboxed"] is True
    assert cfg["EnableTsot"] is False
    assert cfg["DisableCgroup"] is False
    assert "ShimMode" not in cfg


def test_deploy_config_script_valid_json():
    cfg = cri_bench_config_json()
    script = deploy_config_script(cfg)
    assert "/etc/quark/config.json" in script
    start = script.index("{")
    end = script.rindex("}") + 1
    parsed = json.loads(script[start:end])
    assert parsed["Sandboxed"] is True


def test_gate_layers_and_runtime_flag():
    assert set(LAYER_SCRIPTS) == {"L1", "L2", "L3", "L5"}
    assert LAYER_TIMEOUT["L2"] >= LAYER_TIMEOUT["L1"]
    assert runtime_flag("quark") == ""
    assert runtime_flag("kata") == "kata"


def test_bisect_chain_covers_cri_files():
    ids = [b.bug_id for b in BUG_CHAIN]
    assert ids[0] == "032"
    assert ids[-1] == "043"
    assert len(ids) == len(set(ids))
    assert "qvisor/src/main.rs" in all_cri_files()
    assert "qvisor/src/runc/container/container.rs" in all_cri_files()


def test_bisect_032_local_check():
    ok, msg = check_032_local()
    assert ok, msg
