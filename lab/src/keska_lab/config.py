"""Lab configuration — env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabConfig:
    host: str = "lab.keska.vpn"
    user: str = "lab"
    ssh_key: Path | None = None
    repo_path: str = "~/Quark"
    local_repo: Path | None = None
    work_dir: str = "/tmp/keska-lab"
    io_bench_dir: str = "/var/lib/keska-lab/io-bench"
    default_backend: str = "quark"
    bench_iterations: int = 100
    stress_wave_size: int = 20
    stress_waves: int = 5
    bench_image: str = "busybox"
    quark_build_profile: str = "release"
    quark_exec_mode: str = "direct"
    install_prefix: str = "/usr/local"
    sudo_password: str | None = None
    docker_runtime_quark: str = "quark"
    docker_runtime_kata: str = "kata"
    kata_ctr_runtime: str = "io.containerd.kata.v2"
    kata_hypervisor: str = "firecracker"
    enable_tsot: bool = False
    image_registry: str = "europe-north1-docker.pkg.dev/keska-devops/base-images"
    cargo_features: str = ""
    skip_registry_auth: bool = False

    @property
    def quark_bin_dir(self) -> str:
        return f"{self.install_prefix.rstrip('/')}/bin"

    @property
    def quark_binary(self) -> str:
        return "quark_d" if self.quark_build_profile == "debug" else "quark"

    def bundle_path(self, image: str | None = None) -> str:
        from keska_lab.setup.oci_bundle import bundle_dir

        return bundle_dir(self, image or self.bench_image)

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def remote_repo(self) -> str:
        return self.repo_path.replace("~", f"/home/{self.user}")

    def resolve_local_repo(self) -> Path:
        if self.local_repo:
            return self.local_repo.expanduser().resolve()
        env = os.environ.get("KESKA_LAB_LOCAL_REPO")
        if env:
            return Path(env).expanduser().resolve()
        # lab/ → repo root (../../ from lab package)
        return Path(__file__).resolve().parents[3]

    @classmethod
    def from_env(cls) -> LabConfig:
        key = os.environ.get("KESKA_LAB_SSH_KEY")
        local = os.environ.get("KESKA_LAB_LOCAL_REPO")
        return cls(
            host=os.environ.get("KESKA_LAB_HOST", "lab.keska.vpn"),
            user=os.environ.get("KESKA_LAB_USER", "lab"),
            ssh_key=Path(key).expanduser() if key else None,
            repo_path=os.environ.get("KESKA_LAB_REPO", "~/Quark"),
            local_repo=Path(local).expanduser() if local else None,
            work_dir=os.environ.get("KESKA_LAB_WORK", "/tmp/keska-lab"),
            io_bench_dir=os.environ.get("KESKA_LAB_IO_BENCH_DIR", "/var/lib/keska-lab/io-bench"),
            default_backend=os.environ.get("KESKA_LAB_BACKEND", "quark"),
            bench_iterations=int(os.environ.get("KESKA_LAB_BENCH_N", "100")),
            stress_wave_size=int(os.environ.get("KESKA_LAB_STRESS_WAVE", "20")),
            stress_waves=int(os.environ.get("KESKA_LAB_STRESS_WAVES", "5")),
            bench_image=os.environ.get("KESKA_LAB_IMAGE", "busybox"),
            quark_build_profile=os.environ.get("KESKA_LAB_QUARK_PROFILE", "release"),
            quark_exec_mode=os.environ.get("KESKA_LAB_QUARK_EXEC", "direct"),
            install_prefix=os.environ.get("KESKA_LAB_INSTALL_PREFIX", "/usr/local"),
            sudo_password=os.environ.get("KESKA_LAB_SUDO_PASSWORD"),
            docker_runtime_quark=os.environ.get("KESKA_LAB_DOCKER_RUNTIME_QUARK", "quark"),
            kata_hypervisor=os.environ.get("KESKA_LAB_KATA_HYPERVISOR", "firecracker"),
            enable_tsot=os.environ.get("KESKA_LAB_ENABLE_TSOT", "").lower() in ("1", "true", "yes"),
            image_registry=os.environ.get(
                "KESKA_LAB_IMAGE_REGISTRY",
                "europe-north1-docker.pkg.dev/keska-devops/base-images",
            ),
            cargo_features=os.environ.get("KESKA_LAB_CARGO_FEATURES", ""),
            skip_registry_auth=os.environ.get("KESKA_LAB_SKIP_REGISTRY_AUTH", "").lower()
            in ("1", "true", "yes"),
        )

    @property
    def kata_snapshotter(self) -> str | None:
        """containerd snapshotter for Kata (Firecracker needs devmapper)."""
        return "devmapper" if self.kata_hypervisor == "firecracker" else None


@dataclass
class BenchProfile:
    name: str
    sequential: bool = True
    iterations: int = 100
    wave_size: int = 1
    waves: int = 1
    stagger_ms: int = 0

    @classmethod
    def tti(cls, n: int = 100) -> BenchProfile:
        return cls(name="tti", sequential=True, iterations=n, wave_size=1, waves=n)

    @classmethod
    def stress(cls, wave_size: int = 20, waves: int = 5, stagger_ms: int = 50) -> BenchProfile:
        total = wave_size * waves
        return cls(
            name="stress",
            sequential=False,
            iterations=total,
            wave_size=wave_size,
            waves=waves,
            stagger_ms=stagger_ms,
        )

    @classmethod
    def calm(cls, n: int = 10) -> BenchProfile:
        return cls(name="calm", sequential=True, iterations=n, wave_size=1, waves=n)
