"""Backend protocol for sandbox runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from keska_lab.remote import RemoteHost


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


@dataclass
class SandboxHandle:
    backend: str
    sandbox_id: str
    meta: dict


class SandboxBackend(ABC):
    name: str

    def __init__(self, remote: RemoteHost):
        self.remote = remote

    @abstractmethod
    def probe(self) -> dict:
        """Return availability info for this backend on the lab host."""

    @abstractmethod
    def tti_once(self, *, image: str = "busybox") -> float:
        """
        One sequential TTI sample in milliseconds.
        Create/provision → first successful command → stop clock → teardown.

        Quark and Kata (Firecracker) must use the same lifecycle shape so
        comparisons reflect runtime performance, not harness artifacts.
        Teardown is never included in the timed window.
        """

    @abstractmethod
    def stress_once(self, *, image: str = "busybox", wave_index: int = 0) -> float:
        """One parallel cold-start sample in milliseconds (may run concurrently on remote)."""

    def cleanup(self) -> None:
        """Best-effort cleanup of orphaned resources."""
