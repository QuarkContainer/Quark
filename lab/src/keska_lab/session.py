"""Lab session — entry point for IPython shell."""

from __future__ import annotations

from keska_lab.backends import BACKENDS, get_backend
from keska_lab.bench.report import BenchReport
from keska_lab.config import LabConfig
from keska_lab.display import print_suite_compare
from keska_lab.harness.suites import SUITE_MODES
from keska_lab.installer import InstallOptions, NodeInstaller
from keska_lab.knode import KNode
from keska_lab.profile import NodeProfile
from keska_lab.remote import RemoteHost
from keska_lab.runtime import KataEnvironment, QuarkEnvironment, RuntimeEnvironment


class LabSession:
    """
    Central object in the IPython lab shell.

        node = lab.install()           # profile-driven install
        node.bench("network")
        lab.quark.run()                # legacy path
    """

    def __init__(self, profile: NodeProfile | None = None, config: LabConfig | None = None):
        self.config = config or LabConfig.from_env()
        if profile is not None:
            self._profile = profile
            self.config.network_mode = profile.network
        else:
            self._profile = self.config.node_profile
        self.remote = RemoteHost(self.config)
        self.last_report: BenchReport | None = None
        self._node: KNode | None = None
        self._quark: QuarkEnvironment | None = None
        self._kata: KataEnvironment | None = None

    @property
    def profile(self) -> NodeProfile:
        return self._profile

    @property
    def node(self) -> KNode:
        if self._node is None:
            raise RuntimeError("call lab.install() first")
        return self._node

    def install(
        self,
        *,
        profile: NodeProfile | None = None,
        gate_level: str = "L1",
        network_only: bool = False,
    ) -> KNode:
        prof = profile or self._profile
        prof.validate()
        self._profile = prof
        opts = InstallOptions(gate_level=gate_level, network_only=network_only)
        self._node = NodeInstaller(self.remote, prof).install(options=opts)
        return self._node

    def cleanup(self, *, profile: NodeProfile | None = None) -> None:
        prof = profile or self._profile
        NodeInstaller(self.remote, prof).cleanup()
        self._node = None

    @property
    def quark(self) -> QuarkEnvironment:
        if self._quark is None:
            self._quark = QuarkEnvironment(self.remote, self.config)
        return self._quark

    @property
    def kata(self) -> KataEnvironment:
        if self._kata is None:
            self._kata = KataEnvironment(self.remote, self.config)
        return self._kata

    @property
    def backend(self):
        if self._quark and self._quark.last_report:
            return self._quark.backend
        if self._kata and self._kata.last_report:
            return self._kata.backend
        return self.quark.backend

    def ping(self) -> str:
        r = self.remote.ping()
        r.raise_if_failed("lab ping")
        return r.stdout.strip()

    def probe(self) -> dict:
        out = {"host": self.config.ssh_target, "profile": self._profile.name, "backends": {}}
        for name in BACKENDS:
            out["backends"][name] = get_backend(name, self.remote).probe()
        out["docker"] = self.remote.which("docker")
        if self._node is not None:
            out["node"] = {
                "vm_count": self._node.running_vms().__len__(),
                "health_ok": self._node.verify().ok,
            }
        return out

    def use(self, backend: str) -> RuntimeEnvironment:
        if backend == "quark":
            return self.quark
        if backend == "kata":
            return self.kata
        raise ValueError(f"unknown backend {backend!r}; use quark or kata")

    def bench(self, mode: str = "tti", backend: str = "quark", **kwargs):
        if self._node is not None and backend == self._profile.runtime:
            report = self._node.bench(mode, **kwargs)
        else:
            report = self.use(backend).bench(mode, **kwargs)
        if mode not in SUITE_MODES:
            self.last_report = report
        return report

    def bench_all(self, mode: str = "light", *, n: int = 10, setup: bool = True, **kwargs) -> dict:
        out = {}
        for name in ("quark", "kata"):
            out[name] = self.use(name).bench(mode, n=n, setup=setup, **kwargs)
        return out

    def compare(self, results: dict, *, metric: str = "p50") -> None:
        if "quark" not in results or "kata" not in results:
            raise ValueError("results must contain 'quark' and 'kata' keys")
        print_suite_compare(results["quark"], results["kata"])

    def cleanup_sandboxes(self) -> None:
        self.quark.cleanup()
        self.kata.cleanup()

    def ssh(self, cmd: str) -> str:
        return self.remote.sh(cmd, check=True).stdout

    def run(self, cmd: str) -> str:
        return self.ssh(cmd)
