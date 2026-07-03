"""Lab session — entry point for IPython shell."""

from __future__ import annotations

from keska_lab.backends import BACKENDS, get_backend
from keska_lab.bench.report import BenchReport
from keska_lab.config import LabConfig
from keska_lab.display import print_suite_compare
from keska_lab.harness.suites import SUITE_MODES
from keska_lab.remote import RemoteHost
from keska_lab.runtime import KataEnvironment, QuarkEnvironment, RuntimeEnvironment


class LabSession:
    """
    Central object in the IPython lab shell.

        lab.quark.run()              build + install on lab
        lab.quark.bench("light", n=5)
        lab.bench_all("light", n=5)
        lab.compare(results)
    """

    def __init__(self, config: LabConfig | None = None):
        self.config = config or LabConfig.from_env()
        self.remote = RemoteHost(self.config)
        self.last_report: BenchReport | None = None
        self._quark: QuarkEnvironment | None = None
        self._kata: KataEnvironment | None = None

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
        """Active backend of the last-used environment (compat)."""
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
        out = {"host": self.config.ssh_target, "backends": {}}
        for name in BACKENDS:
            out["backends"][name] = get_backend(name, self.remote).probe()
        out["docker"] = self.remote.which("docker")
        return out

    def use(self, backend: str) -> RuntimeEnvironment:
        if backend == "quark":
            return self.quark
        if backend == "kata":
            return self.kata
        raise ValueError(f"unknown backend {backend!r}; use quark or kata")

    def bench(self, mode: str = "tti", backend: str = "quark", **kwargs):
        report = self.use(backend).bench(mode, **kwargs)
        if mode not in SUITE_MODES:
            self.last_report = report
        return report

    def bench_all(self, mode: str = "light", *, n: int = 10, setup: bool = True, **kwargs) -> dict:
        """Run the same benchmark on Quark and Kata; returns {backend: report}."""
        out = {}
        for name in ("quark", "kata"):
            out[name] = self.use(name).bench(mode, n=n, setup=setup, **kwargs)
        return out

    def compare(self, results: dict, *, metric: str = "p50") -> None:
        """Side-by-side p50 ratio table for quark vs kata suite reports."""
        if "quark" not in results or "kata" not in results:
            raise ValueError("results must contain 'quark' and 'kata' keys")
        print_suite_compare(results["quark"], results["kata"])

    def cleanup(self) -> None:
        self.quark.cleanup()
        self.kata.cleanup()

    def ssh(self, cmd: str) -> str:
        """Run a shell command on the lab host."""
        return self.remote.sh(cmd, check=True).stdout

    def run(self, cmd: str) -> str:
        """Alias for ssh — deprecated."""
        return self.ssh(cmd)
