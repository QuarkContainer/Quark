"""Kata Containers backend — containerd ctr (no dockerd in hot path)."""

from __future__ import annotations

import shlex
import textwrap

from keska_lab.backends.base import SandboxBackend
from keska_lab.config import LabConfig
from keska_lab.harness.db import parse_pgbench_tps
from keska_lab.harness.network import (
    INET_CONNECT_PY,
    INET_DOWNLOAD_PY,
    crictl_iperf_script,
    crictl_python_exec_script,
    parse_dd_mib_s,
    parse_iperf_mbps,
)
from keska_lab.harness.workload import is_postgres_workload
from keska_lab.setup.image_registry import ctr_image_ref
from keska_lab.setup.quark_cleanup import cleanup_quark_sandboxes


class KataBackend(SandboxBackend):
    name = "kata"

    def __init__(self, remote):
        super().__init__(remote)

    @property
    def config(self) -> LabConfig:
        return self.remote.config

    @property
    def ctr_runtime(self) -> str:
        return self.config.kata_ctr_runtime

    @property
    def ctr_snapshotter(self) -> str | None:
        return self.config.kata_snapshotter

    def _ctr_snapshotter_flag(self) -> str:
        snap = self.ctr_snapshotter
        return f" --snapshotter {shlex.quote(snap)}" if snap else ""

    def _image_ref(self, image: str) -> str:
        return ctr_image_ref(image)

    def probe(self, *, image: str | None = None) -> dict:
        ctr = self.remote.which("ctr")
        kata = self.remote.which("kata-runtime")
        crictl = self.remote.which("crictl")
        check_image = image or self.config.bench_image
        image_ok = False
        network_ready = False
        if ctr:
            ref = self._image_ref(check_image)
            r = self.remote.sh(
                f"sudo -n ctr images ls name | grep -F {shlex.quote(ref.split(':')[0])} || true",
                timeout=30,
            )
            image_ok = bool(r.stdout.strip())
        if crictl and kata:
            cri = self.remote.sh(
                "test -S /run/containerd/containerd.sock && "
                "sudo -n ctr plugins ls 2>/dev/null | grep -F 'io.containerd.grpc.v1' | grep -w 'cri' | grep -q ' ok ' && "
                "sudo -n crictl info >/dev/null 2>&1 && "
                "sudo -n grep -q 'io.containerd.kata.v2' /etc/containerd/config.toml",
                timeout=30,
            )
            network_ready = cri.ok
        return {
            "ctr": ctr,
            "kata_runtime": kata,
            "crictl": crictl,
            "ctr_image_ready": image_ok,
            "tsot_ready": False,
            "network_ready": network_ready,
            "ready": bool(ctr and kata and image_ok),
        }

    def _ctr_env_flags(self, image: str) -> str:
        if is_postgres_workload(image):
            return (
                " --env POSTGRES_PASSWORD=bench"
                " --env POSTGRES_HOST_AUTH_METHOD=trust"
                " --env PGDATA=/var/lib/postgresql/data"
            )
        return ""

    def _postgres_tti_script(self, *, image: str, probe: str, idle_cmd: str, wait_secs: int = 600) -> str:
        ref = self._image_ref(image)
        env = self._ctr_env_flags(image)
        return textwrap.dedent(
            f"""
            set -euo pipefail
            ID=keska-kata-pg-tti-$RANDOM
            t0=$(date +%s%N)
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {shlex.quote(ref)} "$ID" {idle_cmd} >/dev/null
            ready=0
            for i in $(seq 1 {wait_secs}); do
              if sudo -n ctr task exec --exec-id chk-$RANDOM "$ID" {probe} >/dev/null 2>&1; then
                ready=1
                break
              fi
              sleep 1
            done
            test "$ready" = 1
            t1=$(date +%s%N)
            sudo -n ctr task kill -s SIGKILL "$ID" >/dev/null 2>&1 || true
            sudo -n ctr containers rm "$ID" >/dev/null 2>&1 || true
            echo $(( (t1 - t0) / 1000000 ))
            """
        ).strip()

    def _ctr_run_cmd(self, image: str, cmd: str, *, id_prefix: str = "keska-kata") -> str:
        ref = self._image_ref(image)
        return textwrap.dedent(
            f"""
            set -euo pipefail
            ID={id_prefix}-$RANDOM
            t0=$(date +%s%N)
            sudo -n ctr run --rm --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()} \\
              {shlex.quote(ref)} "$ID" {cmd} >/dev/null
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            """
        ).strip()

    def tti_once(
        self,
        *,
        image: str = "busybox",
        exec_cmd: str = "/bin/echo ok",
        timeout: int = 300,
    ) -> float:
        info = self.probe(image=image)
        if not info["ready"]:
            raise RuntimeError("Kata not ready. Run: lab.kata.run()")
        if is_postgres_workload(image, exec_cmd):
            script = self._postgres_tti_script(
                image=image,
                probe=exec_cmd,
                idle_cmd="docker-entrypoint.sh postgres",
                wait_secs=timeout,
            )
        else:
            script = self._ctr_run_cmd(image, exec_cmd)
        r = self.remote.sh(script, timeout=timeout, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def stress_once(self, *, image: str = "busybox", wave_index: int = 0) -> float:
        del wave_index
        r = self.remote.sh(
            self._ctr_run_cmd(image, "/bin/true", id_prefix="keska-kata-stress"),
            timeout=120,
            check=False,
        )
        if not r.ok:
            raise RuntimeError(r.stderr.strip() or "kata stress sample failed")
        return float(r.stdout.strip().splitlines()[-1])

    def _rss_for_sandbox(self, id_shell_var: str = "$ID") -> str:
        """Awk snippet: sum RSS (MB) for processes tied to a sandbox ID."""
        return (
            f"ID_VAL={id_shell_var}\n"
            f'rss=$(ps -eo rss,args | awk -v id="$ID_VAL" '
            "'index($0, id) {s+=$1} END {printf \"%.2f\", s/1024}')"
        )

    def memory_idle_once(self, *, image: str = "busybox", idle_cmd: str = "/bin/sleep 600") -> float:
        ref = self._image_ref(image)
        env = self._ctr_env_flags(image)
        pg_wait = ""
        if is_postgres_workload(image):
            pg_wait = textwrap.dedent(
                """
                for i in $(seq 1 120); do
                  sudo -n ctr task exec --exec-id chk-$RANDOM "$ID" pg_isready -U postgres >/dev/null 2>&1 && break || sleep 1
                done
                """
            ).strip()
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            ID=keska-kata-mem-$RANDOM
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {shlex.quote(ref)} "$ID" {idle_cmd} >/dev/null
            {pg_wait}
            sleep 2
            {self._rss_for_sandbox("$ID")}
            sudo -n ctr task kill -s SIGKILL "$ID" >/dev/null 2>&1 || true
            sudo -n ctr containers rm "$ID" >/dev/null 2>&1 || true
            echo "${{rss:-0}}"
            """
        ).strip()
        r = self.remote.sh(script, timeout=120, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def pause_resume_once(
        self,
        *,
        image: str = "busybox",
        idle_cmd: str = "/bin/sleep 600",
    ) -> tuple[float, float, float]:
        ref = self._image_ref(image)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            ID=keska-kata-pause-$RANDOM
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()} \\
              {shlex.quote(ref)} "$ID" {idle_cmd} >/dev/null
            sleep 1
            t0=$(date +%s%N)
            sudo -n ctr task pause "$ID" >/dev/null 2>&1
            t1=$(date +%s%N)
            sleep 1
            {self._rss_for_sandbox("$ID")}
            t2=$(date +%s%N)
            sudo -n ctr task resume "$ID" >/dev/null 2>&1
            t3=$(date +%s%N)
            sudo -n ctr task kill -s SIGKILL "$ID" >/dev/null 2>&1 || true
            sudo -n ctr containers rm "$ID" >/dev/null 2>&1 || true
            pause_ms=$(( (t1 - t0) / 1000000 ))
            resume_ms=$(( (t3 - t2) / 1000000 ))
            printf '%s %s %s\\n' "$pause_ms" "$resume_ms" "$rss"
            """
        ).strip()
        r = self.remote.sh(script, timeout=120, check=True)
        parts = r.stdout.strip().splitlines()[-1].split()
        return float(parts[0]), float(parts[1]), float(parts[2])

    def tti_under_load_once(
        self,
        *,
        image: str = "busybox",
        load: int = 4,
        exec_cmd: str = "/bin/echo ok",
    ) -> float:
        ref = self._image_ref(image)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            LOAD_IDS=""
            cleanup() {{
              for id in $LOAD_IDS; do
                sudo -n ctr task kill -s SIGKILL "$id" >/dev/null 2>&1 || true
                sudo -n ctr containers rm "$id" >/dev/null 2>&1 || true
              done
            }}
            trap cleanup EXIT INT TERM
            for i in $(seq 1 {load}); do
              lid=keska-kata-load-$RANDOM-$i
              LOAD_IDS="$LOAD_IDS $lid"
              sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()} \\
                {shlex.quote(ref)} "$lid" /bin/sleep 20 >/dev/null
            done
            sleep 1
            t0=$(date +%s%N)
            ID=keska-kata-tti-$RANDOM
            sudo -n ctr run --rm --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()} \\
              {shlex.quote(ref)} "$ID" {exec_cmd} >/dev/null
            t1=$(date +%s%N)
            cleanup
            echo $(( (t1 - t0) / 1000000 ))
            """
        ).strip()
        r = self.remote.sh(script, timeout=360, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def cleanup(self) -> None:
        cleanup_quark_sandboxes(self.remote)
        self.remote.sh(
            "sudo -n ctr containers ls -q 2>/dev/null | grep '^keska-kata' | "
            "xargs -r -I{} sh -c 'sudo -n ctr task kill -s SIGKILL {} 2>/dev/null; sudo -n ctr containers rm {} 2>/dev/null' || true",
            timeout=30,
        )
        self.remote.sh("sudo -n killall -9 firecracker 2>/dev/null || true", timeout=15)

    def io_fs_once(self, *, image: str = "busybox") -> tuple[float, float]:
        ref = self._image_ref(image)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            ID=keska-kata-io-$RANDOM
            out=$(sudo -n ctr run --rm --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()} \\
              {shlex.quote(ref)} "$ID" sh -c \\
              '"w=$(dd if=/dev/zero of=/tmp/bench bs=1M count=64 conv=fsync 2>&1 | tail -1); '
              'r=$(dd if=/tmp/bench of=/dev/null bs=1M 2>&1 | tail -1); echo WRITE \\"$w\\"; echo READ \\"$r\\""' 2>&1)
            echo "$out"
            """
        ).strip()
        r = self.remote.sh(script, timeout=180, check=True)
        write_mib_s = read_mib_s = 0.0
        for line in r.stdout.splitlines():
            if line.startswith("WRITE "):
                write_mib_s = parse_dd_mib_s(line)
            elif line.startswith("READ "):
                read_mib_s = parse_dd_mib_s(line)
        return write_mib_s, read_mib_s

    def inet_tcp_connect_once_ms(self, *, image: str = "python:3.12-slim") -> float:
        if not self.probe().get("network_ready"):
            raise RuntimeError("Kata CRI network not ready. Run: lab.kata.bench('network', setup=True)")
        script = crictl_python_exec_script(
            INET_CONNECT_PY,
            image=image,
            runtime_handler="kata",
            image_registry=self.config.image_registry,
        )
        r = self.remote.sh(script, timeout=90, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def inet_download_mbps_once(self, *, image: str = "python:3.12-slim") -> float:
        if not self.probe().get("network_ready"):
            raise RuntimeError("Kata CRI network not ready. Run: lab.kata.bench('network', setup=True)")
        script = crictl_python_exec_script(
            INET_DOWNLOAD_PY,
            image=image,
            runtime_handler="kata",
            image_registry=self.config.image_registry,
        )
        r = self.remote.sh(script, timeout=150, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def sandbox_iperf_mbps_once(self, *, image: str = "networkstatic/iperf3") -> float:
        if not self.probe().get("network_ready"):
            raise RuntimeError("Kata CRI network not ready. Run: lab.kata.bench('network', setup=True)")
        script = crictl_iperf_script(
            image=image,
            runtime_handler="kata",
            image_registry=self.config.image_registry,
        )
        r = self.remote.sh(script, timeout=180, check=True)
        mbps = parse_iperf_mbps(r.stdout)
        if mbps is None:
            raise RuntimeError("no iperf throughput parsed")
        return mbps

    def pgbench_tps_once(
        self,
        *,
        image: str = "postgres:16-alpine",
        idle_cmd: str = "docker-entrypoint.sh postgres",
    ) -> float:
        ref = self._image_ref(image)
        env = self._ctr_env_flags(image)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            ID=keska-kata-pg-$RANDOM
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {shlex.quote(ref)} "$ID" {idle_cmd} >/dev/null
            for i in $(seq 1 120); do
              sudo -n ctr task exec --exec-id chk-$RANDOM "$ID" pg_isready -U postgres >/dev/null 2>&1 && break || sleep 1
            done
            sudo -n ctr task exec --user 70:70 --exec-id init-$RANDOM "$ID" pgbench -i -s1 -U postgres >/dev/null
            out=$(sudo -n ctr task exec --user 70:70 --exec-id pgb-$RANDOM "$ID" pgbench -c1 -T5 -U postgres 2>&1 || true)
            sudo -n ctr task kill -s SIGKILL "$ID" >/dev/null 2>&1 || true
            sudo -n ctr containers rm "$ID" >/dev/null 2>&1 || true
            echo "$out"
            """
        ).strip()
        r = self.remote.sh(script, timeout=300, check=True)
        return parse_pgbench_tps(r.stdout)
