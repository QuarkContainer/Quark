"""Benchmark orchestration."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from keska_lab.backends.base import SandboxBackend
from keska_lab.bench.report import BenchReport
from keska_lab.config import BenchProfile, LabConfig
from keska_lab.display import print_bench_report


def run_benchmark(
    backend: SandboxBackend,
    profile: BenchProfile,
    *,
    config: LabConfig | None = None,
    image: str = "busybox",
    verbose: bool = False,
) -> BenchReport:
    cfg = config or LabConfig.from_env()
    samples: list[float] = []
    errors: list[str] = []

    backend.cleanup()

    if profile.sequential:
        batch_fn = getattr(backend, "tti_batch", None)
        if profile.name == "tti" and callable(batch_fn):
            try:
                samples = batch_fn(profile.iterations, image=image)
                if verbose:
                    for i, ms in enumerate(samples, 1):
                        print(f"  [{i}/{profile.iterations}] {ms:.1f} ms")
            except Exception as e:
                errors.append(str(e))
                if verbose:
                    print(f"  batch ERROR: {e}")
        else:
            for i in range(profile.iterations):
                try:
                    ms = backend.tti_once(image=image)
                    samples.append(ms)
                    if verbose:
                        print(f"  [{i + 1}/{profile.iterations}] {ms:.1f} ms")
                except Exception as e:
                    errors.append(str(e))
                    if verbose:
                        print(f"  [{i + 1}/{profile.iterations}] ERROR: {e}")
    else:
        for wave in range(profile.waves):
            wave_samples: list[float] = []
            with ThreadPoolExecutor(max_workers=profile.wave_size) as pool:
                futures = [
                    pool.submit(backend.stress_once, image=image, wave_index=wave)
                    for _ in range(profile.wave_size)
                ]
                for fut in as_completed(futures):
                    try:
                        wave_samples.append(fut.result())
                    except Exception as e:
                        errors.append(str(e))
            samples.extend(wave_samples)
            if verbose:
                print(f"  wave {wave + 1}/{profile.waves}: {len(wave_samples)} ok")
            if profile.stagger_ms:
                time.sleep(profile.stagger_ms / 1000.0)

    report = BenchReport.build(
        backend=backend.name,
        profile=profile.name,
        host=cfg.ssh_target,
        samples=samples,
        errors=errors,
    )
    backend.cleanup()
    return report


def bench_and_print(
    backend: SandboxBackend,
    profile: BenchProfile,
    *,
    save_dir: str | None = None,
    verbose: bool = False,
    image: str = "busybox",
) -> BenchReport:
    report = run_benchmark(backend, profile, verbose=verbose, image=image)
    print_bench_report(report)
    if save_dir:
        from pathlib import Path

        ts = report.timestamp.replace(":", "-")
        path = Path(save_dir) / f"{ts}_{report.backend}_{report.profile}.json"
        report.save(path)
        print(f"Saved {path}")
    return report
