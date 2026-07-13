"""Benchmark suite with optional setup phase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from keska_lab.bench.report import BenchReport
from keska_lab.bench.runner import bench_and_print
from keska_lab.config import BenchProfile, LabConfig
from keska_lab.harness.suites import SUITE_MODES
from keska_lab.harness import SuiteReport, run_suite

if TYPE_CHECKING:
    from keska_lab.runtime import RuntimeEnvironment


@dataclass
class BenchmarkSuite:
    """
    Runs setup → benchmark → optional cleanup for a RuntimeEnvironment.

        lab.quark.bench("light", setup=True)
        lab.quark.bench("full")  # alias for light
    """

    environment: RuntimeEnvironment
    mode: str = "tti"
    n: int | None = None
    wave_size: int | None = None
    waves: int | None = None
    image: str = "busybox"
    workload: str | None = None
    setup: bool = True
    verbose: bool = True
    save: bool = True
    profile: str | None = None
    exec_mode: str | None = None

    def configure(
        self,
        *,
        mode: str = "tti",
        n: int | None = None,
        wave_size: int | None = None,
        waves: int | None = None,
        image: str | None = None,
        workload: str | None = None,
        setup: bool = True,
        verbose: bool = True,
        save: bool = True,
        profile: str | None = None,
        exec_mode: str | None = None,
    ) -> BenchmarkSuite:
        self.mode = mode
        self.n = n
        self.wave_size = wave_size
        self.waves = waves
        if image:
            self.image = image
        if workload:
            self.workload = workload
        self.setup = setup
        self.verbose = verbose
        self.save = save
        self.profile = profile
        self.exec_mode = exec_mode
        return self

    def _profile(self, config: LabConfig) -> BenchProfile:
        if self.mode == "tti":
            return BenchProfile.tti(self.n or config.bench_iterations)
        if self.mode == "calm":
            return BenchProfile.calm(self.n or 10)
        if self.mode == "stress":
            return BenchProfile.stress(
                self.wave_size or config.stress_wave_size,
                self.waves or config.stress_waves,
            )
        if self.mode in SUITE_MODES:
            return BenchProfile.calm(self.n or 10)
        raise ValueError(f"unknown mode {self.mode!r}")

    def run(self) -> BenchReport | Any:
        env = self.environment
        if self.mode in SUITE_MODES:
            return self._run_suite(env)
        if self.setup:
            env.prepare(mode=self.mode, workload=self.workload)

        profile = self._profile(env.config)
        save_dir = None
        if self.save:
            save_dir = str(Path.home() / ".keska-lab" / "results")

        report = bench_and_print(
            env.backend,
            profile,
            save_dir=save_dir,
            verbose=self.verbose,
            image=self.image,
        )
        env.last_report = report
        return report

    def _run_suite(self, env) -> SuiteReport | dict[str, SuiteReport]:
        from keska_lab.display import print_suite_report

        if self.setup:
            env.prepare(mode=self.mode, workload=self.workload)

        n = self.n or 10
        suite = self.mode

        if suite == "standard":
            reports = {}
            for wl in ("busybox", "python"):
                rep = run_suite(env.backend, suite="light" if wl == "busybox" else "workloads", workload=wl, n=n, verbose=self.verbose)
                reports[wl] = rep
                print_suite_report(rep)
                if self.save:
                    self._save_report(rep, suffix=wl)
            env.last_suite_reports = reports
            env.backend.cleanup()
            return reports

        workload = self.workload or ("postgres" if suite == "db" else "busybox")
        if suite == "network" and env.name == "quark":
            from keska_lab.profile import NetworkMode
            from keska_lab.setup.quark_config import (
                cri_tsot_bench_config_json,
                deploy_config_script,
            )

            if env.config.network_mode == NetworkMode.tsot:
                r = env.remote.sh(deploy_config_script(cri_tsot_bench_config_json()), timeout=60)
                if not r.ok:
                    raise RuntimeError(f"TSOT CRI config deploy failed: {env.remote.format_failure(r)}")
        report = run_suite(
            env.backend,
            suite=suite,
            workload=workload,
            n=n,
            verbose=self.verbose,
        )
        if suite == "network" and env.name == "quark":
            from keska_lab.setup.quark_config import QuarkBenchConfigStep

            QuarkBenchConfigStep().run(env.remote)
        print_suite_report(report)
        if self.save:
            self._save_report(report)
        env.last_suite_report = report
        return report

    def _save_report(self, report: SuiteReport, *, suffix: str = "") -> None:
        ts = report.timestamp.replace(":", "-")
        tag = f"_{suffix}" if suffix else ""
        path = Path.home() / ".keska-lab" / "results" / (
            f"{ts}_{report.backend}_{report.suite}{tag}.json"
        )
        report.save(path)
        if self.verbose:
            print(f"Saved {path}")
