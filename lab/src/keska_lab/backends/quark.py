"""Quark sandbox backend — direct OCI CLI (default) or optional Docker runtime."""

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
    python_exec_cmd,
)
from keska_lab.harness.workload import is_postgres_workload
from keska_lab.setup.oci_bundle import bundle_dir, postgres_data_template_dir
from keska_lab.setup.quark_cleanup import cleanup_quark_sandboxes
from keska_lab.setup.tsot import TSOT_SOCKET, tsot_ready_script


class QuarkBackend(SandboxBackend):
    name = "quark"

    def __init__(
        self,
        remote,
        *,
        profile: str | None = None,
        exec_mode: str | None = None,
    ):
        super().__init__(remote)
        self.profile = profile or self.config.quark_build_profile
        self.exec_mode = exec_mode or self.config.quark_exec_mode

    @property
    def config(self) -> LabConfig:
        return self.remote.config

    @property
    def quark_bin(self) -> str:
        return "quark_d" if self.profile == "debug" else "quark"

    @property
    def docker_runtime(self) -> str:
        return self.quark_bin if self.profile == "debug" else self.config.docker_runtime_quark

    def probe(self, *, image: str | None = None) -> dict:
        quark = self.remote.which(self.quark_bin)
        check_image = image or self.config.bench_image
        bundle = bundle_dir(self.config, check_image)
        bundle_ok = False
        if quark:
            r = self.remote.sh(
                f"test -f {shlex.quote(bundle)}/config.json && test -d {shlex.quote(bundle)}/rootfs && echo OK",
                timeout=15,
            )
            bundle_ok = "OK" in r.stdout
        docker = self.remote.which("docker")
        docker_runtime = False
        if docker and self.exec_mode == "docker":
            r = self.remote.docker_sh("docker info 2>/dev/null | grep -i quark || true", timeout=30)
            docker_runtime = "quark" in r.stdout.lower()
        tsot = False
        if quark:
            tr = self.remote.sh(tsot_ready_script(), timeout=15)
            tsot = tr.ok
        return {
            "quark_binary": quark,
            "profile": self.profile,
            "exec_mode": self.exec_mode,
            "bundle_ready": bundle_ok,
            "docker": docker,
            "docker_quark_runtime": docker_runtime,
            "tsot_ready": tsot,
            "network_ready": tsot,
            "ready": bool(quark and (bundle_ok or self.exec_mode == "docker")),
        }

    def _bundle_path(self, image: str) -> str:
        return bundle_dir(self.config, image)

    def _quark_cmd(self, subcmd: str) -> str:
        bin_path = f"{self.config.quark_bin_dir}/{self.quark_bin}"
        return f"sudo -n {shlex.quote(bin_path)} {subcmd}"

    def _quark_force_delete(self, id_ref: str = '"$ID"') -> str:
        """Delete sandbox with timeout; kill orphaned VM if delete hangs."""
        qdel = f"timeout 20 {self._quark_cmd(f'delete --force {id_ref}')}"
        return (
            f"({qdel} >/dev/null 2>&1) || "
            f"{{ sudo -n pkill -9 -f '[q]uark boot' 2>/dev/null || true; "
            f"sudo -n pkill -9 qemu 2>/dev/null || true; "
            f"timeout 15 {self._quark_cmd(f'delete --force {id_ref}')} >/dev/null 2>&1 || true; }}"
        )

    def _pg_ready_wait_secs(self, timeout: int) -> int:
        return min(timeout, 60)

    def _postgres_exec_user(self) -> str:
        return "--user 70:70"

    def _postgres_startup_sleep(self) -> str:
        """Wait for postgres init without quark exec (exec stops daemon sandboxes)."""
        return "sleep 3"

    def _postgres_run_bundle_script(self, *, base_bundle_var: str = "BUNDLE") -> str:
        base = bundle_dir(self.config, "postgres:16-alpine")
        tmpl = postgres_data_template_dir(self.config)
        return textwrap.dedent(
            f"""
            BASE={shlex.quote(base)}
            TMPL={shlex.quote(tmpl)}
            {base_bundle_var}=/tmp/keska-lab/pg-run-$RANDOM
            DATA="${base_bundle_var}/data"
            mkdir -p "$DATA"
            if ! mountpoint -q "$DATA" 2>/dev/null; then
              sudo -n mount -t tmpfs -o size=512m,mode=1777 tmpfs "$DATA"
            fi
            sudo cp -a "$TMPL/." "$DATA/"
            sudo chown -R 70:70 "$DATA"
            sudo chmod 700 "$DATA"
            ln -sfn "$BASE/rootfs" "${base_bundle_var}/rootfs"
            BASE="$BASE" BUNDLE="${base_bundle_var}" python3 -c "import json, os, pathlib; base=pathlib.Path(os.environ['BASE']); bundle=pathlib.Path(os.environ['BUNDLE']); data=bundle/'data'; cfg=json.loads((base/'config.json').read_text()); cfg['mounts']=[m for m in cfg.get('mounts', []) if m.get('destination')!='/var/lib/postgresql/data']; cfg['mounts'].append({{'destination':'/var/lib/postgresql/data','type':'bind','source':str(data),'options':['rbind','rw']}}); (bundle/'config.json').write_text(json.dumps(cfg, indent=2)+'\\n')"
            trap 'sudo -n umount "$DATA" 2>/dev/null || true' EXIT INT TERM
            """
        ).strip()

    def _bundle_setup_for_image(self, image: str, *, bundle_var: str = "BUNDLE") -> str:
        if not is_postgres_workload(image):
            return f'{bundle_var}={shlex.quote(self._bundle_path(image))}'
        return self._postgres_run_bundle_script(base_bundle_var=bundle_var)

    def _postgres_tti_script(self, *, probe: str, wait_secs: int = 600) -> str:
        q = self._quark_cmd
        prep = self._postgres_run_bundle_script()
        return textwrap.dedent(
            f"""
            set -euo pipefail
            {prep}
            ID=keska-pg-tti-$RANDOM
            cleanup() {{ {self._quark_force_delete()}; sudo rm -rf "$BUNDLE" 2>/dev/null || true; }}
            trap cleanup EXIT INT TERM
            t0=$(date +%s%N)
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            ready=0
            for i in $(seq 1 {wait_secs}); do
              if {q(f'exec --user 0:0 "$ID" -- {probe}')} >/dev/null 2>&1; then
                ready=1
                break
              fi
              sleep 1
            done
            test "$ready" = 1
            t1=$(date +%s%N)
            cleanup
            echo $(( (t1 - t0) / 1000000 ))
            """
        ).strip()

    def _direct_lifecycle_script(
        self,
        *,
        bundle: str,
        id_prefix: str = "keska",
        exec_cmd: str = "/bin/echo ok",
        timed: bool = True,
    ) -> str:
        q = self._quark_cmd
        timing = ""
        if timed:
            timing = textwrap.dedent(
                """
                t0=$(date +%s%N)
                """
            ).strip()
            end = textwrap.dedent(
                """
                t1=$(date +%s%N)
                echo $(( (t1 - t0) / 1000000 ))
                """
            ).strip()
        else:
            end = ""
        return textwrap.dedent(
            f"""
            set -euo pipefail
            ID={id_prefix}-$RANDOM
            BUNDLE={shlex.quote(bundle)}
            cleanup() {{
              {q(f'delete --force "$ID"')} >/dev/null 2>&1 || true
            }}
            trap cleanup EXIT INT TERM
            {timing}
            {q(f'create "$ID" -b "$BUNDLE"')}
            {q(f'start "$ID"')}
            {q(f'exec --user 0:0 "$ID" -- {exec_cmd}')} >/dev/null
            cleanup
            trap - EXIT INT TERM
            {end}
            """
        ).strip()

    def _docker_tti_script(self, image: str, cmd: str) -> str:
        return textwrap.dedent(
            f"""
            set -euo pipefail
            t0=$(date +%s%N)
            sg docker -c 'docker run --runtime={self.docker_runtime!r} --rm {image!r} {cmd} >/dev/null'
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
            raise RuntimeError(
                f"Quark not ready ({self.quark_bin}, exec_mode={self.exec_mode}). "
                "Run: lab.quark.run() or lab.quark.prepare()"
            )
        if self.exec_mode == "docker":
            script = self._docker_tti_script(image, exec_cmd)
        elif is_postgres_workload(image, exec_cmd):
            wait = self._pg_ready_wait_secs(timeout)
            script = self._postgres_tti_script(
                probe=exec_cmd,
                wait_secs=wait,
            )
            timeout = min(timeout, wait + 90)
        else:
            script = self._direct_lifecycle_script(
                bundle=self._bundle_path(image),
                exec_cmd=exec_cmd,
            )
        r = self.remote.sh(script, timeout=timeout, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def stress_once(self, *, image: str = "busybox", wave_index: int = 0) -> float:
        del wave_index
        if self.exec_mode == "docker":
            script = self._docker_tti_script(image, "/bin/true")
        else:
            script = self._direct_lifecycle_script(
                bundle=self._bundle_path(image),
                id_prefix="keska-stress",
                exec_cmd="/bin/true",
            )
        r = self.remote.sh(script, timeout=120, check=False)
        if not r.ok:
            raise RuntimeError(r.stderr.strip() or "stress sample failed")
        return float(r.stdout.strip().splitlines()[-1])

    def memory_idle_once(self, *, image: str = "busybox", idle_cmd: str = "/bin/sleep 600") -> float:
        q = self._quark_cmd
        bundle_setup = self._bundle_setup_for_image(image)
        pg_wait = self._postgres_startup_sleep() if is_postgres_workload(image) else ""
        cleanup_rm = 'sudo rm -rf "$BUNDLE" 2>/dev/null || true' if is_postgres_workload(image) else ""
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            {bundle_setup}
            ID=keska-mem-$RANDOM
            cleanup() {{
              {self._quark_force_delete()}
              {cleanup_rm}
            }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            {pg_wait}
            sleep 2
            rss=$(ps -eo rss,comm | awk '$2 ~ /quark|qvisor|qemu|cloud-hypervisor|virtiofsd/ {{s+=$1}} END {{printf "%.2f", s/1024}}')
            cleanup
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
        q = self._quark_cmd
        bundle = self._bundle_path(image)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            ID=keska-pause-$RANDOM
            BUNDLE={shlex.quote(bundle)}
            cleanup() {{ {q('delete --force "$ID"')} >/dev/null 2>&1 || true; }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            sleep 1
            t0=$(date +%s%N)
            {q('pause "$ID"')} >/dev/null 2>&1
            t1=$(date +%s%N)
            sleep 1
            rss=$(ps -eo rss,comm | awk '$2 ~ /quark|qvisor|qemu|cloud-hypervisor|virtiofsd/ {{s+=$1}} END {{printf "%.2f", s/1024}}')
            t2=$(date +%s%N)
            {q('resume "$ID"')} >/dev/null 2>&1
            t3=$(date +%s%N)
            cleanup
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
        q = self._quark_cmd
        bundle = self._bundle_path(image)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            BUNDLE={shlex.quote(bundle)}
            LOAD_IDS=""
            cleanup() {{
              for id in $LOAD_IDS; do
                {q('delete --force "$id"')} >/dev/null 2>&1 || true
              done
              {q('delete --force "$ID"')} >/dev/null 2>&1 || true
            }}
            trap cleanup EXIT INT TERM
            for i in $(seq 1 {load}); do
              lid=keska-load-$RANDOM-$i
              LOAD_IDS="$LOAD_IDS $lid"
              {q('create "$lid" -b "$BUNDLE"')} >/dev/null
              {q('start "$lid"')} >/dev/null
            done
            sleep 1
            ID=keska-tti-$RANDOM
            t0=$(date +%s%N)
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            {q(f'exec --user 0:0 "$ID" -- {exec_cmd}')} >/dev/null
            t1=$(date +%s%N)
            cleanup
            echo $(( (t1 - t0) / 1000000 ))
            """
        ).strip()
        r = self.remote.sh(script, timeout=360, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def cleanup(self) -> None:
        cleanup_quark_sandboxes(self.remote)

    def _exec_in_sandbox(self, sandbox_id: str, cmd: str, *, timeout: int = 120) -> str:
        q = self._quark_cmd
        r = self.remote.sh(
            f"{q(f'exec --user 0:0 {shlex.quote(sandbox_id)} -- {cmd}')}",
            timeout=timeout,
            check=True,
        )
        return r.stdout.strip()

    def _sandbox_ipv4(self, sandbox_id: str) -> str:
        out = self._exec_in_sandbox(
            sandbox_id,
            "ip -4 -o addr show scope global | awk '{print $4}' | head -1 | cut -d/ -f1",
        )
        line = out.splitlines()[-1].strip() if out else ""
        if not line or line == "127.0.0.1":
            raise RuntimeError(f"no pod IPv4 for sandbox {sandbox_id}")
        return line

    def io_fs_once(self, *, image: str = "busybox") -> tuple[float, float]:
        q = self._quark_cmd
        bundle = self._bundle_path(image)
        io_exec = q(
            'exec --user 0:0 "$ID" -- sh -c '
            '"w=$(dd if=/dev/zero of=/tmp/bench bs=1M count=64 conv=fsync 2>&1 | tail -1); '
            'r=$(dd if=/tmp/bench of=/dev/null bs=1M 2>&1 | tail -1); '
            'echo WRITE \\"$w\\"; echo READ \\"$r\\""'
        )
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            ID=keska-io-$RANDOM
            BUNDLE={shlex.quote(bundle)}
            cleanup() {{ {q('delete --force "$ID"')} >/dev/null 2>&1 || true; }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            out=$({io_exec})
            cleanup
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

    def _python_exec_sample(
        self,
        code: str,
        *,
        image: str = "python:3.12-slim",
        timeout: int = 120,
    ) -> float:
        py = python_exec_cmd(code)
        q = self._quark_cmd
        bundle = self._bundle_path(image)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            ID=keska-py-$RANDOM
            BUNDLE={shlex.quote(bundle)}
            cleanup() {{ {q('delete --force "$ID"')} >/dev/null 2>&1 || true; }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            val=$({q('exec --user 0:0 "$ID" --')} {py})
            cleanup
            echo "$val" | tail -1
            """
        ).strip()
        r = self.remote.sh(script, timeout=timeout, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def _crictl_python_exec_sample(
        self,
        code: str,
        *,
        image: str = "python:3.12-slim",
        timeout: int = 120,
    ) -> float:
        script = crictl_python_exec_script(
            code,
            image=image,
            runtime=self.docker_runtime,
            image_registry=self.config.image_registry,
        )
        r = self.remote.sh(script, timeout=timeout, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def _crictl_iperf_sample(self, *, image: str = "networkstatic/iperf3") -> float:
        script = crictl_iperf_script(
            image=image,
            runtime=self.docker_runtime,
            image_registry=self.config.image_registry,
        )
        r = self.remote.sh(script, timeout=180, check=True)
        mbps = parse_iperf_mbps(r.stdout)
        if mbps is None:
            raise RuntimeError("no iperf throughput parsed")
        return mbps

    def _network_exec_mode(self) -> str:
        if self.probe().get("tsot_ready"):
            return "crictl"
        return self.exec_mode

    def inet_tcp_connect_once_ms(self, *, image: str = "python:3.12-slim") -> float:
        if not self.probe().get("tsot_ready"):
            raise RuntimeError(f"TSOT not ready ({TSOT_SOCKET})")
        if self._network_exec_mode() == "crictl":
            return self._crictl_python_exec_sample(INET_CONNECT_PY, image=image)
        return self._python_exec_sample(INET_CONNECT_PY, image=image)

    def inet_download_mbps_once(self, *, image: str = "python:3.12-slim") -> float:
        if not self.probe().get("tsot_ready"):
            raise RuntimeError(f"TSOT not ready ({TSOT_SOCKET})")
        if self._network_exec_mode() == "crictl":
            return self._crictl_python_exec_sample(INET_DOWNLOAD_PY, image=image, timeout=120)
        return self._python_exec_sample(INET_DOWNLOAD_PY, image=image, timeout=120)

    def sandbox_iperf_mbps_once(self, *, image: str = "networkstatic/iperf3") -> float:
        if not self.probe().get("tsot_ready"):
            raise RuntimeError(f"TSOT not ready ({TSOT_SOCKET})")
        if self._network_exec_mode() == "crictl":
            return self._crictl_iperf_sample(image=image)
        q = self._quark_cmd
        server_bundle = self._bundle_path(image)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            SRV=keska-iperf-s-$RANDOM
            CLI=keska-iperf-c-$RANDOM
            BUNDLE={shlex.quote(server_bundle)}
            cleanup() {{
              {q('delete --force "$SRV"')} >/dev/null 2>&1 || true
              {q('delete --force "$CLI"')} >/dev/null 2>&1 || true
            }}
            trap cleanup EXIT INT TERM
            {q('create "$SRV" -b "$BUNDLE"')}
            {q('start "$SRV"')}
            {q('exec --user 0:0 "$SRV" -- sh -c')} "iperf3 -s -D && sleep 2"
            IP=$({q('exec --user 0:0 "$SRV" -- ip -4 -o addr show scope global')} \\
              | awk '{{print $4}}' | head -1 | cut -d/ -f1)
            test -n "$IP"
            {q('create "$CLI" -b "$BUNDLE"')}
            {q('start "$CLI"')}
            out=$({q('exec --user 0:0 "$CLI" -- iperf3 -c "$IP" -t 5 -f m')} 2>&1 || true)
            cleanup
            echo "$out"
            """
        ).strip()
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
        del idle_cmd
        q = self._quark_cmd
        prep = self._postgres_run_bundle_script()
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            {prep}
            ID=keska-pg-$RANDOM
            cleanup() {{ {self._quark_force_delete()}; sudo rm -rf "$BUNDLE" 2>/dev/null || true; }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            {self._postgres_startup_sleep()}
            out=$(timeout 30 {q(f'exec {self._postgres_exec_user()} "$ID" -- pgbench -c1 -T5 -U postgres')} 2>&1) || true
            test -n "$out"
            cleanup
            echo "$out"
            """
        ).strip()
        r = self.remote.sh(script, timeout=120, check=True)
        return parse_pgbench_tps(r.stdout)
