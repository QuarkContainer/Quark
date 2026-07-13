"""Unit tests for install pipelines and config scripts."""

from __future__ import annotations

from keska_lab.installer import InstallOptions, NodeInstaller
from keska_lab.profile import NetworkMode, NodeProfile
from keska_lab.setup.cni import cni_conflist_type_script, tsot_conflist_body
from keska_lab.setup.quark_config import bench_config_json, quark_config_enable_tsot_script
from keska_lab.setup.containerd_cri import containerd_cri_config_script


def test_bench_config_tsot_from_profile():
    cfg = bench_config_json(NodeProfile.quark_tsot())
    assert cfg["EnableTsot"] is True


def test_bench_config_bridge_from_profile():
    cfg = bench_config_json(NodeProfile.quark_bridge())
    assert cfg["EnableTsot"] is False


def test_quark_config_check_script():
    s = quark_config_enable_tsot_script(True)
    assert "EnableTsot" in s
    assert "True" in s


def test_tsot_conflist_type():
    body = tsot_conflist_body()
    assert '"type": "tsot"' in body


def test_install_pipeline_includes_tsot_for_quark_tsot():
    from keska_lab.config import LabConfig
    from keska_lab.remote import RemoteHost

    prof = NodeProfile.quark_tsot()
    inst = NodeInstaller(RemoteHost(LabConfig()), prof)
    names = [s.name for s in inst._install_pipeline(InstallOptions())._steps]
    assert "cni-tsot" in names
    assert "tsot-stack" in names


def test_install_pipeline_bridge_has_no_tsot_stack():
    from keska_lab.config import LabConfig
    from keska_lab.remote import RemoteHost

    prof = NodeProfile.quark_bridge()
    inst = NodeInstaller(RemoteHost(LabConfig()), prof)
    names = [s.name for s in inst._install_pipeline(InstallOptions())._steps]
    assert "cni-bridge" in names
    assert "tsot-stack" not in names
    assert names.index("tsot-stack-stop") < names.index("cni-bridge")


def test_install_pipeline_kata_stops_tsot_before_cni():
    from keska_lab.config import LabConfig
    from keska_lab.remote import RemoteHost

    prof = NodeProfile.kata_bridge()
    inst = NodeInstaller(RemoteHost(LabConfig()), prof)
    names = [s.name for s in inst._install_pipeline(InstallOptions())._steps]
    assert names.index("tsot-stack-stop") < names.index("cni-bridge")
    assert "tsot-stack" not in names


def test_install_pipeline_tsot_defers_containerd_lifecycle_smoke():
    from keska_lab.config import LabConfig
    from keska_lab.remote import RemoteHost

    prof = NodeProfile.quark_tsot()
    inst = NodeInstaller(RemoteHost(LabConfig()), prof)
    names = [s.name for s in inst._install_pipeline(InstallOptions())._steps]
    assert names.index("containerd-cri") < names.index("tsot-stack")


def test_install_pipeline_network_only_skips_provision():
    from keska_lab.config import LabConfig
    from keska_lab.remote import RemoteHost

    prof = NodeProfile.quark_tsot()
    inst = NodeInstaller(RemoteHost(LabConfig()), prof)
    names = [s.name for s in inst._install_pipeline(InstallOptions(network_only=True))._steps]
    assert "provision-quark" not in names
    assert "containerd-cri" not in names
    assert "quark-config" in names
    assert "cni-tsot" in names
    assert "tsot-stack" in names


def test_containerd_config_quark_omits_kata_devmapper():
    script = containerd_cri_config_script(include_kata=False)
    assert "INCLUDE_KATA = False" in script
    assert "if INCLUDE_KATA:" in script
    assert "if INCLUDE_KATA and not re.search(" in script


def test_containerd_config_kata_includes_devmapper():
    script = containerd_cri_config_script(include_kata=True)
    assert "INCLUDE_KATA = True" in script
    assert "snapshotter = 'devmapper'" in script


def test_cni_type_script_has_bridge_check():
    s = cni_conflist_type_script("bridge")
    assert "bridge" in s
    assert "exit 1" in s
