"""Sandbox backends."""

from keska_lab.backends.base import ExecResult, SandboxBackend, SandboxHandle
from keska_lab.backends.kata import KataBackend
from keska_lab.backends.quark import QuarkBackend

BACKENDS: dict[str, type[SandboxBackend]] = {
    "quark": QuarkBackend,
    "kata": KataBackend,
}


def get_backend(
    name: str,
    remote,
    *,
    profile: str | None = None,
    exec_mode: str | None = None,
) -> SandboxBackend:
    cls = BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"unknown backend {name!r}; choose from {list(BACKENDS)}")
    if name == "quark":
        return cls(remote, profile=profile, exec_mode=exec_mode)
    return cls(remote)
