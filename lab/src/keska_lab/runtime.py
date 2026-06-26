"""Runtime environments — backend + setup + bench."""

from __future__ import annotations

from dataclasses import dataclass, field

from keska_lab.backends.base import SandboxBackend
from keska_lab.backends import get_backend
from keska_lab.bench.report import BenchReport
from keska_lab.bench.suite import BenchmarkSuite
from keska_lab.config import LabConfig
from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupReport
from keska_lab.harness.suites import SUITE_MODES
from keska_lab.setup.pipelines import (
    kata_bench_ready_pipeline,
    quark_db_ready_pipeline,
    quark_network_ready_pipeline,
    workload_setup_pipeline,
)


@dataclass
class RuntimeEnvironment:
    name: str
    remote: RemoteHost
    config: LabConfig
    backend: SandboxBackend
    last_setup: SetupReport | None = None
    last_report: BenchReport | None = None
    _results_dir: str = field(
        default_factory=lambda: str(__import__("pathlib").Path.home() / ".keska-lab" / "results")
    )

    def probe(self) -> dict:
        return self.backend.probe()

    def bench(
        self,
        mode: str = "tti",
        *,
        setup: bool = True,
        n: int | None = None,
        wave_size: int | None = None,
        waves: int | None = None,
        image: str | None = None,
        workload: str | None = None,
        verbose: bool = True,
        save: bool = True,
        profile: str | None = None,
        exec_mode: str | None = None,
    ):
        if self.name == "quark" and (profile or exec_mode):
            from keska_lab.backends import get_backend

            self.backend = get_backend(
                "quark",
                self.remote,
                profile=profile or self.config.quark_build_profile,
                exec_mode=exec_mode or self.config.quark_exec_mode,
            )
        suite = BenchmarkSuite(self).configure(
            mode=mode,
            n=n,
            wave_size=wave_size,
            waves=waves,
            image=image or self.config.bench_image,
            workload=workload,
            setup=setup,
            verbose=verbose,
            save=save,
            profile=profile,
            exec_mode=exec_mode,
        )
        report = suite.run()
        if mode not in SUITE_MODES:
            self.last_report = report
        return report

    def cleanup(self) -> None:
        self.backend.cleanup()


class QuarkEnvironment(RuntimeEnvironment):
    """Quark runtime on the lab host."""

    def __init__(self, remote: RemoteHost, config: LabConfig):
        super().__init__(
            name="quark",
            remote=remote,
            config=config,
            backend=get_backend(
                "quark",
                remote,
                profile=config.quark_build_profile,
                exec_mode=config.quark_exec_mode,
            ),
        )

    def run(self, *, stream: bool = True) -> SetupReport:
        """Build and install Quark on lab: lab.quark.run()"""
        from keska_lab.display import console, print_setup_report
        from keska_lab.provision import provision_quark

        report = provision_quark(self.remote, self.config, stream=stream)
        self.last_setup = report
        print_setup_report(report)
        if not report.ok:
            failed = next(s for s in report.steps if not s.ok)
            if "\n" in failed.message:
                console.print(f"\n[red]{failed.name}[/red]")
                console.print(failed.message)
            report.raise_if_failed()
        return report

    def prepare(
        self,
        *,
        stream: bool = False,
        mode: str | None = None,
        workload: str | None = None,
    ) -> SetupReport:
        from keska_lab.display import print_setup_report

        wl = workload or "busybox"
        if mode in ("standard", "heavy"):
            extras = ["python"] if wl == "busybox" else []
            report = workload_setup_pipeline(self.config, wl, extra_workloads=extras).run(
                self.remote, stream=stream
            )
        elif mode == "network":
            report = quark_network_ready_pipeline(self.config, workload=wl).run(
                self.remote, stream=stream
            )
        elif mode == "db":
            report = quark_db_ready_pipeline(self.config).run(self.remote, stream=stream)
        else:
            report = workload_setup_pipeline(self.config, wl).run(self.remote, stream=stream)
        self.last_setup = report
        print_setup_report(report)
        if not report.ok:
            report.raise_if_failed()
        return report


class KataEnvironment(RuntimeEnvironment):
    """Kata Containers via containerd ctr (no dockerd in hot path)."""

    def __init__(self, remote: RemoteHost, config: LabConfig):
        super().__init__(
            name="kata",
            remote=remote,
            config=config,
            backend=get_backend("kata", remote),
        )

    def run(self, *, stream: bool = False) -> SetupReport:
        from keska_lab.display import print_setup_report

        report = kata_bench_ready_pipeline(self.config).run(self.remote, stream=stream)
        self.last_setup = report
        print_setup_report(report)
        if not report.ok:
            report.raise_if_failed()
        return report

    def prepare(
        self,
        *,
        stream: bool = False,
        mode: str | None = None,
        workload: str | None = None,
    ) -> SetupReport:
        from keska_lab.display import print_setup_report
        from keska_lab.setup.pipelines import (
            kata_bench_ready_pipeline,
            kata_db_ready_pipeline,
            kata_multi_image_pipeline,
            kata_network_ready_pipeline,
        )

        if mode == "network":
            report = kata_network_ready_pipeline(self.config, workload=workload).run(
                self.remote, stream=stream
            )
        elif mode == "db":
            report = kata_db_ready_pipeline(self.config).run(self.remote, stream=stream)
        elif mode in ("standard", "heavy"):
            report = kata_multi_image_pipeline(self.config, ["busybox", "python"]).run(
                self.remote, stream=stream
            )
        else:
            report = kata_bench_ready_pipeline(self.config, workload or "busybox").run(
                self.remote, stream=stream
            )
        self.last_setup = report
        print_setup_report(report)
        if not report.ok:
            report.raise_if_failed()
        return report
