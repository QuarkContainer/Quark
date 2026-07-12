"""Host node configuration — pure data, no remote calls."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ProfileError(ValueError):
    pass


class NetworkMode(str, Enum):
    bridge = "bridge"
    tsot = "tsot"
    rdma = "rdma"


@dataclass(frozen=True)
class NetworkParams:
    cidr: str = "10.1.1.0/8"
    node_ip: str = "127.0.0.1"
    pod_mgr_port: int = 8888
    tsot_cni_port: int = 1234
    tsot_svc_port: int = 1235
    state_svc_port: int = 8890


@dataclass(frozen=True)
class NodeProfile:
    runtime: Literal["quark", "kata"]
    network: NetworkMode
    build_profile: str = "release"
    bench_image: str = "busybox"
    net_params: NetworkParams = field(default_factory=NetworkParams)

    def validate(self) -> None:
        if self.network == NetworkMode.rdma:
            raise ProfileError("network mode 'rdma' is not implemented yet")
        if self.network == NetworkMode.tsot and self.runtime != "quark":
            raise ProfileError("tsot network mode requires runtime=quark")

    @property
    def name(self) -> str:
        return f"{self.runtime}_{self.network.value}"

    @classmethod
    def quark_bridge(cls, **kwargs) -> NodeProfile:
        return cls(runtime="quark", network=NetworkMode.bridge, **kwargs)

    @classmethod
    def quark_tsot(cls, **kwargs) -> NodeProfile:
        return cls(runtime="quark", network=NetworkMode.tsot, **kwargs)

    @classmethod
    def kata_bridge(cls, **kwargs) -> NodeProfile:
        return cls(runtime="kata", network=NetworkMode.bridge, **kwargs)

    @classmethod
    def from_lab_config(cls, config) -> NodeProfile:
        """Build from LabConfig (compat bridge)."""
        runtime = config.default_backend if config.default_backend in ("quark", "kata") else "quark"
        network = config.network_mode
        if config.enable_tsot and network == NetworkMode.bridge:
            network = NetworkMode.tsot
        return cls(
            runtime=runtime,
            network=network,
            build_profile=config.quark_build_profile,
            bench_image=config.bench_image,
        )

    @classmethod
    def from_env(cls) -> NodeProfile:
        runtime = os.environ.get("KESKA_LAB_RUNTIME", os.environ.get("KESKA_LAB_BACKEND", "quark"))
        if runtime not in ("quark", "kata"):
            runtime = "quark"

        mode_str = os.environ.get("KESKA_LAB_NETWORK_MODE", "").strip().lower()
        legacy_tsot = os.environ.get("KESKA_LAB_ENABLE_TSOT", "").lower() in ("1", "true", "yes")
        if legacy_tsot and not mode_str:
            warnings.warn(
                "KESKA_LAB_ENABLE_TSOT is deprecated; use KESKA_LAB_NETWORK_MODE=tsot",
                DeprecationWarning,
                stacklevel=2,
            )
            mode_str = "tsot"
        if not mode_str:
            mode_str = "bridge"

        try:
            network = NetworkMode(mode_str)
        except ValueError as e:
            raise ProfileError(
                f"invalid KESKA_LAB_NETWORK_MODE={mode_str!r}; use bridge, tsot, or rdma"
            ) from e

        profile = cls(
            runtime=runtime,  # type: ignore[arg-type]
            network=network,
            build_profile=os.environ.get("KESKA_LAB_QUARK_PROFILE", "release"),
            bench_image=os.environ.get("KESKA_LAB_IMAGE", "busybox"),
        )
        profile.validate()
        return profile
