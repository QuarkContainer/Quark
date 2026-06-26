"""Pre-built setup pipelines for Quark and Kata benchmarks."""

from __future__ import annotations

from keska_lab.config import LabConfig
from keska_lab.harness.workload import get_workload
from keska_lab.remote import RemoteHost
from keska_lab.setup.base import SetupPipeline, SetupStep, StepResult
from keska_lab.setup.cni import CniPluginsStep
from keska_lab.setup.containerd_cri import ContainerdCriStep, CrictlInstallStep
from keska_lab.setup.docker import (
    DockerEnsureStep,
    DockerPullStep,
    DockerRuntimeStep,
    KataInstallStep,
    QuarkRuntimeCheckStep,
)
from keska_lab.setup.image_registry import ImageRegistryAuthStep
from keska_lab.setup.kata_firecracker import KataHypervisorStep
from keska_lab.setup.oci_bundle import (
    CtrImagePullStep,
    EnsureWorkloadBundleStep,
)
from keska_lab.setup.quark_cleanup import cleanup_quark_sandboxes
from keska_lab.setup.quark_config import QuarkBenchConfigStep
from keska_lab.setup.tsot import TsotBenchReadyStep


class CleanupSandboxesStep(SetupStep):
    name = "cleanup-sandboxes"

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        cleanup_quark_sandboxes(remote)
        return StepResult(self.name, True, "stopped stray sandboxes")


class QuarkDirectCheckStep(SetupStep):
    name = "quark-direct-check"

    def __init__(self, config: LabConfig, workload: str = "busybox"):
        self.config = config
        self.workload = workload

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        import shlex

        from keska_lab.backends.quark import QuarkBackend
        from keska_lab.setup.oci_bundle import bundle_dir

        spec = get_workload(self.workload)
        backend = QuarkBackend(remote, profile=self.config.quark_build_profile, exec_mode="direct")
        if not remote.which(self.config.quark_binary):
            return StepResult(self.name, False, f"{self.config.quark_binary} not installed")
        bundle = bundle_dir(self.config, spec.image)
        br = remote.sh(
            f"test -f {shlex.quote(bundle)}/config.json && test -d {shlex.quote(bundle)}/rootfs && echo OK",
            timeout=15,
        )
        if not br.ok or "OK" not in br.stdout:
            return StepResult(self.name, False, f"bundle not ready: {bundle}")
        try:
            ms = backend.tti_once(image=spec.image, exec_cmd=spec.tti_exec, timeout=spec.tti_timeout)
            return StepResult(self.name, True, f"direct OCI smoke OK ({ms:.0f} ms)")
        except Exception as e:
            return StepResult(self.name, False, str(e))


class KataCtrCheckStep(SetupStep):
    name = "kata-ctr-check"

    def __init__(self, config: LabConfig, workload: str = "busybox"):
        self.config = config
        self.workload = workload

    def run(self, remote: RemoteHost, *, stream: bool = False) -> StepResult:
        from keska_lab.backends.kata import KataBackend

        spec = get_workload(self.workload)
        backend = KataBackend(remote)
        info = backend.probe(image=spec.image)
        if not info["ready"]:
            return StepResult(self.name, False, f"kata ctr not ready: {info}")
        try:
            ms = backend.tti_once(
                image=spec.image,
                exec_cmd=spec.tti_exec,
                timeout=spec.tti_timeout,
            )
            return StepResult(self.name, True, f"ctr smoke OK ({ms:.0f} ms)")
        except Exception as e:
            return StepResult(self.name, False, str(e))


def _workloads_for_mode(mode: str | None, workload: str | None) -> list[str]:
    if workload:
        return [workload]
    if mode in ("standard", "heavy"):
        return ["busybox", "python"]
    if mode == "db":
        return ["postgres"]
    if mode == "network":
        return ["python", "iperf"]
    return ["busybox"]


def _lab_prep(cfg: LabConfig) -> list[SetupStep]:
    return [CleanupSandboxesStep(), DockerEnsureStep(), ImageRegistryAuthStep(cfg)]


def workload_setup_pipeline(
    config: LabConfig | None = None,
    workload: str = "busybox",
    *,
    extra_workloads: list[str] | None = None,
) -> SetupPipeline:
    cfg = config or LabConfig.from_env()
    names = list(dict.fromkeys([workload, *(extra_workloads or [])]))
    pipe = SetupPipeline("workload-setup").extend(_lab_prep(cfg)).add(QuarkBenchConfigStep())
    for name in names:
        pipe.add(EnsureWorkloadBundleStep(cfg, name))
    pipe.add(QuarkDirectCheckStep(cfg, workload=names[0]))
    if cfg.quark_exec_mode == "docker":
        pipe.add(DockerRuntimeStep(cfg)).add(QuarkRuntimeCheckStep())
    return pipe


def quark_bench_ready_pipeline(
    config: LabConfig | None = None,
    workload: str = "busybox",
) -> SetupPipeline:
    return workload_setup_pipeline(config, workload)


def quark_network_ready_pipeline(
    config: LabConfig | None = None,
    workload: str | None = None,
) -> SetupPipeline:
    cfg = config or LabConfig.from_env()
    extras = _workloads_for_mode("network", workload)
    pipe = workload_setup_pipeline(cfg, extras[0], extra_workloads=extras[1:])
    pipe.add(CniPluginsStep()).add(ContainerdCriStep()).add(CrictlInstallStep())
    if cfg.enable_tsot:
        pipe.add(TsotBenchReadyStep())
    for name in extras:
        spec = get_workload(name)
        pipe.add(DockerPullStep(spec.image, cfg))
    return pipe


def kata_bench_ready_pipeline(
    config: LabConfig | None = None,
    workload: str = "busybox",
) -> SetupPipeline:
    cfg = config or LabConfig.from_env()
    spec = get_workload(workload)
    return (
        SetupPipeline("kata-bench-ready")
        .extend(_lab_prep(cfg))
        .add(KataInstallStep())
        .add(KataHypervisorStep(cfg))
        .add(CtrImagePullStep(spec.image, cfg))
        .add(KataCtrCheckStep(cfg, workload=workload))
    )


def kata_multi_image_pipeline(config: LabConfig | None = None, workloads: list[str] | None = None) -> SetupPipeline:
    cfg = config or LabConfig.from_env()
    names = workloads or ["busybox", "python"]
    pipe = (
        SetupPipeline("kata-images-ready")
        .extend(_lab_prep(cfg))
        .add(KataInstallStep())
        .add(KataHypervisorStep(cfg))
    )
    for name in names:
        spec = get_workload(name)
        pipe.add(CtrImagePullStep(spec.image, cfg))
    pipe.add(KataCtrCheckStep(cfg, workload=names[0]))
    return pipe


def kata_network_ready_pipeline(
    config: LabConfig | None = None,
    workload: str | None = None,
) -> SetupPipeline:
    cfg = config or LabConfig.from_env()
    extras = _workloads_for_mode("network", workload)
    pipe = (
        SetupPipeline("kata-network-ready")
        .extend(_lab_prep(cfg))
        .add(KataInstallStep())
        .add(KataHypervisorStep(cfg))
        .add(CniPluginsStep())
        .add(ContainerdCriStep())
        .add(CrictlInstallStep())
    )
    for name in extras:
        spec = get_workload(name)
        pipe.add(CtrImagePullStep(spec.image, cfg))
    pipe.add(KataCtrCheckStep(cfg, workload=extras[0]))
    return pipe


def quark_db_ready_pipeline(config: LabConfig | None = None) -> SetupPipeline:
    cfg = config or LabConfig.from_env()
    return (
        SetupPipeline("quark-db-ready")
        .extend(_lab_prep(cfg))
        .add(QuarkBenchConfigStep())
        .add(EnsureWorkloadBundleStep(cfg, "postgres"))
        .add(QuarkDirectCheckStep(cfg, workload="postgres"))
    )


def kata_db_ready_pipeline(config: LabConfig | None = None) -> SetupPipeline:
    cfg = config or LabConfig.from_env()
    spec = get_workload("postgres")
    return (
        SetupPipeline("kata-db-ready")
        .extend(_lab_prep(cfg))
        .add(KataInstallStep())
        .add(KataHypervisorStep(cfg))
        .add(CtrImagePullStep(spec.image, cfg))
        .add(KataCtrCheckStep(cfg, workload="postgres"))
    )
