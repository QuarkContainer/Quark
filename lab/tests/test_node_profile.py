"""Unit tests for NodeProfile."""

from __future__ import annotations

import pytest

from keska_lab.config import LabConfig
from keska_lab.profile import NetworkMode, NodeProfile, ProfileError


def test_quark_tsot_valid():
    p = NodeProfile.quark_tsot()
    p.validate()


def test_kata_tsot_invalid():
    with pytest.raises(ProfileError):
        NodeProfile(runtime="kata", network=NetworkMode.tsot).validate()


def test_from_lab_config_bridge():
    cfg = LabConfig(network_mode=NetworkMode.bridge, default_backend="quark")
    p = NodeProfile.from_lab_config(cfg)
    assert p.network == NetworkMode.bridge
    assert p.runtime == "quark"


def test_from_lab_config_tsot_via_enable_tsot_property():
    cfg = LabConfig(network_mode=NetworkMode.tsot)
    assert cfg.enable_tsot is True
    p = cfg.node_profile
    assert p.network == NetworkMode.tsot
