"""Quark sandbox backend — direct OCI CLI (default) or optional Docker runtime."""

from __future__ import annotations

import shlex
import textwrap

from keska_lab.backends.base import SandboxBackend
from keska_lab.config import LabConfig
from keska_lab.harness.db import parse_pgbench_tps
from keska_lab.harness.batch import remote_batch_loop, remote_batch_script
from keska_lab.harness.metrics import (
    MEMORY_RSS_SETTLE_SECS,
    parse_float_lines,
    parse_metric_samples,
    parse_pause_resume_lines,
    quark_rss_for_sandbox_shell,
    rss_by_args_match_shell,
)
from keska_lab.harness.postgres import POSTGRES_STARTUP_SLEEP_SECS
from keska_lab.harness.io_fs import (
    IO_BENCH_FILE,
    IO_BENCH_MOUNT,
    concurrent_read_bench_sh,
    concurrent_read_mib_s,
    concurrent_read_total_bytes,
    dd_io_bench_sh,
    quark_io_bench_bundle_preamble,
    quark_io_bench_cleanup_trap,
)
from keska_lab.harness.network import (
    INET_CONNECT_PY,
    INET_DOWNLOAD_PY,
    crictl_iperf_script,
    crictl_python_exec_script,
    parse_dd_mib_s,
    parse_iperf_mbps,
    python_exec_cmd,
)
from keska_lab.harness.workload import (
    is_postgres_workload,
    micro_bench_script,
    normalize_cpu_loop_inner,
)
from keska_lab.setup.oci_bundle import (
    bundle_dir,
    postgres_data_template_dir,
    refresh_bundle_rootfs_script,
)
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
        cri_ready = False
        if quark:
            tr = self.remote.sh(tsot_ready_script(), timeout=15)
            tsot = tr.ok
            cri = self.remote.sh(
                "test -S /run/containerd/containerd.sock && "
                "sudo -n crictl info >/dev/null 2>&1",
                timeout=15,
            )
            cri_ready = cri.ok
        return {
            "quark_binary": quark,
            "profile": self.profile,
            "exec_mode": self.exec_mode,
            "bundle_ready": bundle_ok,
            "docker": docker,
            "docker_quark_runtime": docker_runtime,
            "tsot_ready": tsot,
            "cri_ready": cri_ready,
            "network_ready": tsot or cri_ready,
            "ready": bool(quark and (bundle_ok or self.exec_mode == "docker")),
        }

    def _bundle_path(self, image: str) -> str:
        return bundle_dir(self.config, image)

    def _refresh_rootfs(self, image: str, *, bundle_var: str = "BUNDLE") -> str:
        return refresh_bundle_rootfs_script(bundle_var=bundle_var, image=image)

    def _quark_cmd(self, subcmd: str) -> str:
        bin_path = f"{self.config.quark_bin_dir}/{self.quark_bin}"
        return f"sudo -n {shlex.quote(bin_path)} {subcmd}"

    def _quark_force_delete(self, id_ref: str = '"$ID"') -> str:
        """Delete sandbox with timeout; kill its VM if delete hangs."""
        id_shell = id_ref.strip('"')  # e.g. $ID or $id — expanded by bash at runtime
        kill = (
            f'__meta="/run/qvisor/{id_shell}/meta.json"; '
            f'__pid=$(python3 -c "import json,sys; '
            f'd=json.load(open(sys.argv[1])); '
            f"print(d.get('Sandbox',{{}}).get('Pid',-1))\" "
            f'"$__meta" 2>/dev/null || echo -1); '
            f'if [ "$__pid" = "-1" ] || [ "$__pid" = "0" ] || [ -z "$__pid" ]; then '
            f'__pid=""; fi; '
            f'if [ -n "$__pid" ]; then '
            f"sudo -n kill -9 \"$__pid\" 2>/dev/null || true; fi"
        )
        rm_meta = (
            f'sudo rm -rf "/run/qvisor/{id_shell}" "/var/lib/quark/{id_shell}" '
            f"2>/dev/null || true"
        )
        qdel = f"timeout 5 {self._quark_cmd(f'delete --force {id_ref}')}"
        return (
            f"({qdel} >/dev/null 2>&1) || "
            f"{{ {kill}; "
            f"timeout 5 {self._quark_cmd(f'delete --force {id_ref}')} >/dev/null 2>&1 || true; "
            f"{rm_meta}; }}"
        )

    def _quark_timed_create_start(self, id_ref: str = '"$ID"') -> str:
        q = self._quark_cmd
        return (
            f"timeout 30 {q(f'create {id_ref} -b \"$BUNDLE\"')}\n"
            f"timeout 60 {q(f'start {id_ref}')}"
        )

    def _bench_exec_timeout(self, exec_cmd: str) -> int:
        """Seconds for ``timeout`` wrapping ``quark exec`` in harness scripts."""
        if exec_cmd.strip() in {"/bin/echo ok", "/bin/true"}:
            return 15
        return 60

    def _bench_stale_metadata_wipe_shell(self) -> str:
        return (
            "sudo rm -rf /run/qvisor/keska-* /var/lib/quark/keska-* "
            "/var/lib/quark/keska_* /run/qvisor/keska_* 2>/dev/null || true"
        )

    def _pg_ready_wait_secs(self, timeout: int, *, exec_timeout: int = 8) -> int:
        """Loop iterations for pg_isready; budget fits within SSH timeout."""
        overhead = 45  # create, start, cleanup
        per_iter = exec_timeout + 1
        budget = max(10, timeout - overhead)
        return min(60, budget // per_iter)

    def _postgres_exec_user(self) -> str:
        return "--user 70:70"

    def _postgres_startup_sleep(self) -> str:
        """Wait for postgres init without quark exec (exec stops daemon sandboxes)."""
        return f"sleep {POSTGRES_STARTUP_SLEEP_SECS}"

    def _postgres_run_bundle_script(self, *, base_bundle_var: str = "BUNDLE") -> str:
        base = bundle_dir(self.config, "postgres:16-alpine")
        tmpl = postgres_data_template_dir(self.config)
        return textwrap.dedent(
            f"""
            BASE={shlex.quote(base)}
            TMPL={shlex.quote(tmpl)}
            sudo test -f "$TMPL/PG_VERSION" || {{
              echo "missing postgres data template at $TMPL (run workload bundle setup)" >&2
              exit 1
            }}
            {self._refresh_rootfs("postgres:16-alpine", bundle_var="BASE")}
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
            path = self._bundle_path(image)
            return (
                f"{bundle_var}={shlex.quote(path)}\n"
                f"{self._refresh_rootfs(image, bundle_var=bundle_var)}"
            )
        return self._postgres_run_bundle_script(base_bundle_var=bundle_var)

    def _postgres_tti_script(
        self, *, probe: str, wait_secs: int = 600, exec_timeout: int = 8
    ) -> str:
        q = self._quark_cmd
        prep = self._postgres_run_bundle_script()
        exec_user = self._postgres_exec_user()
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
              if timeout {exec_timeout} {q(f'exec {exec_user} "$ID" -- {probe}')} >/tmp/keska-pgprobe-$$.log 2>&1; then
                ready=1
                break
              fi
              sleep 1
            done
            test "$ready" = 1
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            cleanup
            """
        ).strip()

    def _direct_lifecycle_script(
        self,
        *,
        bundle: str,
        image: str,
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
            sudo rm -rf /run/qvisor/keska-* /var/lib/quark/keska-* 2>/dev/null || true
            ID={id_prefix}-$RANDOM
            BUNDLE={shlex.quote(bundle)}
            {self._refresh_rootfs(image, bundle_var="BUNDLE")}
            cleanup() {{
              {self._quark_force_delete()}
            }}
            trap cleanup EXIT INT TERM
            {timing}
            {q(f'create "$ID" -b "$BUNDLE"')}
            {q(f'start "$ID"')}
            timeout 60 {q(f'exec --user 0:0 "$ID" -- {exec_cmd}')}
            {end}
            cleanup
            trap - EXIT INT TERM
            """
        ).strip()

    def _direct_tti_iter_body(
        self,
        *,
        exec_cmd: str,
        id_prefix: str = "keska",
    ) -> str:
        """One timed TTI iteration; expects ``BUNDLE`` to be set by the preamble."""
        q = self._quark_cmd
        return textwrap.dedent(
            f"""
            ID={id_prefix}-$RANDOM
            cleanup() {{ {self._quark_force_delete()}; }}
            trap cleanup EXIT INT TERM
            t0=$(date +%s%N)
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            timeout 60 {q(f'exec --user 0:0 "$ID" -- {exec_cmd}')}
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            cleanup
            trap - EXIT INT TERM
            """
        ).strip()

    def _direct_vm_boot_iter_body(self, *, id_prefix: str = "keska-boot") -> str:
        """One timed create+start iteration; expects ``BUNDLE`` from the preamble."""
        return textwrap.dedent(
            f"""
            ID={id_prefix}-$RANDOM
            t0=$(date +%s%N)
            {self._quark_timed_create_start()}
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            {self._quark_force_delete()}
            """
        ).strip()

    def _bundle_batch_preamble(self, image: str) -> str:
        if is_postgres_workload(image):
            return self._postgres_run_bundle_script()
        path = self._bundle_path(image)
        return (
            f"{self._bench_stale_metadata_wipe_shell()}\n"
            f"BUNDLE={shlex.quote(path)}\n"
            f"{self._refresh_rootfs(image, bundle_var='BUNDLE')}"
        )

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
            exec_timeout = 8
            wait = self._pg_ready_wait_secs(timeout, exec_timeout=exec_timeout)
            script = self._postgres_tti_script(
                probe=exec_cmd,
                wait_secs=wait,
                exec_timeout=exec_timeout,
            )
            timeout = min(timeout, 45 + wait * (exec_timeout + 1) + 30)
        else:
            script = self._direct_lifecycle_script(
                bundle=self._bundle_path(image),
                image=image,
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
                image=image,
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
            sleep {MEMORY_RSS_SETTLE_SECS}
            {quark_rss_for_sandbox_shell(self._quark_cmd("list"))}
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
            cleanup() {{ {self._quark_force_delete()}; }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            sleep {MEMORY_RSS_SETTLE_SECS}
            t0=$(date +%s%N)
            {q('pause "$ID"')} >/dev/null 2>&1
            t1=$(date +%s%N)
            sleep {MEMORY_RSS_SETTLE_SECS}
            {quark_rss_for_sandbox_shell(self._quark_cmd("list"))}
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
            {self._refresh_rootfs(image, bundle_var="BUNDLE")}
            LOAD_IDS=""
            ID=""
            cleanup() {{
              for id in $LOAD_IDS; do
                {self._quark_force_delete('"$id"')}
              done
              if [ -n "${{ID:-}}" ]; then
                {self._quark_force_delete()}
              fi
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
            timeout 60 {q(f'exec --user 0:0 "$ID" -- {exec_cmd}')}
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            cleanup
            """
        ).strip()
        r = self.remote.sh(script, timeout=360, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def _quark_exec_sh_inner(self, id_ref: str, inner_sh: str) -> str:
        q = self._quark_cmd
        return q(f"exec --user 0:0 {id_ref} -- /bin/sh -c {shlex.quote(inner_sh)}")

    def _cpu_loop_body(
        self,
        *,
        image: str,
        inner_sh: str,
        id_prefix: str = "keska-cpu",
    ) -> str:
        q = self._quark_cmd
        bundle_setup = self._bundle_setup_for_image(image)
        cleanup_rm = 'sudo rm -rf "$BUNDLE" 2>/dev/null || true' if is_postgres_workload(image) else ""
        pg_wait = self._postgres_startup_sleep() if is_postgres_workload(image) else ""
        exec_line = self._quark_exec_sh_inner('"$ID"', inner_sh)
        return textwrap.dedent(
            f"""
            {bundle_setup}
            ID={id_prefix}-$RANDOM
            cleanup() {{
              {self._quark_force_delete()}
              {cleanup_rm}
            }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            {pg_wait}
            t0=$(date +%s%N)
            timeout 120 {exec_line} >/dev/null
            t1=$(date +%s%N)
            cleanup
            trap - EXIT INT TERM
            echo $(( (t1 - t0) / 1000000 ))
            """
        ).strip()

    def cpu_loop_once(
        self,
        *,
        image: str = "busybox",
        exec_cmd: str | None = None,
    ) -> float:
        if self.exec_mode == "docker":
            raise NotImplementedError("cpu_loop_once requires direct OCI exec_mode")
        inner = normalize_cpu_loop_inner(exec_cmd)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            {self._cpu_loop_body(image=image, inner_sh=inner)}
            """
        ).strip()
        r = self.remote.sh(script, timeout=180, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def tti_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        exec_cmd: str = "/bin/echo ok",
        timeout: int = 300,
    ) -> list[float]:
        if self.exec_mode == "docker":
            raise NotImplementedError("tti_batch requires direct OCI exec_mode")
        per = min(timeout, 30)
        if is_postgres_workload(image, exec_cmd):
            raise NotImplementedError("tti_batch for postgres not supported")
        script = remote_batch_script(
            preamble=self._bundle_batch_preamble(image),
            n=n,
            body=self._direct_tti_iter_body(exec_cmd=exec_cmd),
        )
        r = self.remote.sh(script, timeout=n * per + 60, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def vm_boot_once(
        self,
        *,
        image: str = "busybox",
        idle_cmd: str = "/bin/sleep 600",
    ) -> float:
        del idle_cmd
        return self.vm_boot_batch(1, image=image)[0]

    def vm_boot_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        idle_cmd: str = "/bin/sleep 600",
    ) -> list[float]:
        del idle_cmd
        if self.exec_mode == "docker":
            raise NotImplementedError("vm_boot_batch requires direct OCI exec_mode")
        if is_postgres_workload(image):
            raise NotImplementedError("vm_boot_batch for postgres not supported")
        script = remote_batch_script(
            preamble=self._bundle_batch_preamble(image),
            n=n,
            body=self._direct_vm_boot_iter_body(),
        )
        r = self.remote.sh(script, timeout=n * 30 + 60, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def memory_idle_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        idle_cmd: str = "/bin/sleep 600",
    ) -> list[float]:
        q = self._quark_cmd
        pg_wait = self._postgres_startup_sleep() if is_postgres_workload(image) else ""
        cleanup_rm = 'sudo rm -rf "$BUNDLE" 2>/dev/null || true' if is_postgres_workload(image) else ""
        iter_body = textwrap.dedent(
            f"""
            ID=keska-mem-$RANDOM
            cleanup() {{
              {self._quark_force_delete()}
              {cleanup_rm}
            }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            {pg_wait}
            sleep {MEMORY_RSS_SETTLE_SECS}
            {quark_rss_for_sandbox_shell(q("list"))}
            cleanup
            trap - EXIT INT TERM
            echo "${{rss:-0}}"
            """
        ).strip()
        script = remote_batch_script(
            preamble=self._bundle_batch_preamble(image),
            n=n,
            body=iter_body,
        )
        r = self.remote.sh(script, timeout=n * 60 + 60, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def pause_resume_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        idle_cmd: str = "/bin/sleep 600",
    ) -> list[tuple[float, float, float]]:
        q = self._quark_cmd
        iter_body = textwrap.dedent(
            f"""
            ID=keska-pause-$RANDOM
            cleanup() {{ {self._quark_force_delete()}; }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            sleep {MEMORY_RSS_SETTLE_SECS}
            t0=$(date +%s%N)
            {q('pause "$ID"')} >/dev/null 2>&1
            t1=$(date +%s%N)
            {quark_rss_for_sandbox_shell(q("list"))}
            t2=$(date +%s%N)
            {q('resume "$ID"')} >/dev/null 2>&1
            t3=$(date +%s%N)
            cleanup
            trap - EXIT INT TERM
            pause_ms=$(( (t1 - t0) / 1000000 ))
            resume_ms=$(( (t3 - t2) / 1000000 ))
            printf '%s %s %s\\n' "$pause_ms" "$resume_ms" "$rss"
            """
        ).strip()
        script = remote_batch_script(
            preamble=self._bundle_batch_preamble(image),
            n=n,
            body=iter_body,
        )
        r = self.remote.sh(script, timeout=n * 60 + 60, check=True)
        return parse_pause_resume_lines(r.stdout, expect=n)

    def _light_suite_batch_script(
        self,
        n: int,
        *,
        image: str,
        exec_cmd: str,
        cpu_loop_inner: str,
        load: int = 4,
    ) -> str:
        q = self._quark_cmd
        qlist = q("list")
        rss = quark_rss_for_sandbox_shell(qlist)
        cpu_exec = self._quark_exec_sh_inner('"$ID"', cpu_loop_inner)
        exec_line = q(f'exec --user 0:0 "$ID" -- {exec_cmd}')
        exec_timeout = self._bench_exec_timeout(exec_cmd)
        settle = MEMORY_RSS_SETTLE_SECS

        tti_body = textwrap.dedent(
            f"""
            ID=keska-tti-$RANDOM
            t0=$(date +%s%N)
            {self._quark_timed_create_start()}
            timeout 60 {exec_line} >/dev/null
            t1=$(date +%s%N)
            echo "METRIC tti_ms SAMPLE $__batch_i $(( (t1 - t0) / 1000000 ))"
            {self._quark_force_delete()}
            """
        ).strip()

        batch_warmup = textwrap.dedent(
            f"""
            ID=keska-warmup-$RANDOM
            {self._quark_timed_create_start()}
            timeout 60 {exec_line} >/dev/null || true
            {self._quark_force_delete()}
            """
        ).strip()

        vm_boot_body = textwrap.dedent(
            f"""
            ID=keska-boot-$RANDOM
            t0=$(date +%s%N)
            {self._quark_timed_create_start()}
            t1=$(date +%s%N)
            echo "METRIC vm_boot_ms SAMPLE $__batch_i $(( (t1 - t0) / 1000000 ))"
            {self._quark_force_delete()}
            """
        ).strip()

        tti_load_body = textwrap.dedent(
            f"""
            ID=keska-tti-$RANDOM
            t0=$(date +%s%N)
            {self._quark_timed_create_start()}
            timeout 60 {exec_line} >/dev/null
            t1=$(date +%s%N)
            echo "METRIC tti_under_load_ms SAMPLE $__batch_i $(( (t1 - t0) / 1000000 ))"
            {self._quark_force_delete()}
            """
        ).strip()

        load_setup = textwrap.dedent(
            f"""
            LOAD_IDS=""
            cleanup_load() {{
              for id in $LOAD_IDS; do
                {self._quark_force_delete('"$id"')}
              done
            }}
            for i in $(seq 1 {load}); do
              lid=keska-load-$RANDOM-$i
              LOAD_IDS="$LOAD_IDS $lid"
              timeout 30 {q('create "$lid" -b "$BUNDLE"')} >/dev/null
              timeout 60 {q('start "$lid"')} >/dev/null
            done
            sleep {settle}
            trap cleanup_load EXIT INT TERM
            """
        ).strip()

        mem_body = textwrap.dedent(
            f"""
            ID=keska-mem-$RANDOM
            cleanup() {{ {self._quark_force_delete()}; }}
            trap cleanup EXIT INT TERM
            {self._quark_timed_create_start()}
            sleep {settle}
            {rss}
            echo "METRIC memory_idle_rss_mb SAMPLE $__batch_i ${{rss:-0}}"
            cleanup
            trap - EXIT INT TERM
            """
        ).strip()

        pause_body = textwrap.dedent(
            f"""
            ID=keska-pause-$RANDOM
            cleanup() {{ {self._quark_force_delete()}; }}
            trap cleanup EXIT INT TERM
            {self._quark_timed_create_start()}
            sleep {settle}
            t0=$(date +%s%N)
            {q('pause "$ID"')} >/dev/null 2>&1
            t1=$(date +%s%N)
            {rss}
            t2=$(date +%s%N)
            {q('resume "$ID"')} >/dev/null 2>&1
            t3=$(date +%s%N)
            pause_ms=$(( (t1 - t0) / 1000000 ))
            resume_ms=$(( (t3 - t2) / 1000000 ))
            echo "METRIC pause_ms SAMPLE $__batch_i $pause_ms"
            echo "METRIC resume_ms SAMPLE $__batch_i $resume_ms"
            echo "METRIC memory_while_paused_rss_mb SAMPLE $__batch_i ${{rss:-0}}"
            cleanup
            trap - EXIT INT TERM
            """
        ).strip()

        cpu_loop_body = textwrap.dedent(
            f"""
            t0=$(date +%s%N)
            timeout 120 {cpu_exec} >/dev/null
            t1=$(date +%s%N)
            echo "METRIC cpu_loop_ms SAMPLE $__batch_i $(( (t1 - t0) / 1000000 ))"
            """
        ).strip()

        cpu_preamble = textwrap.dedent(
            f"""
            ID=keska-cpu-$RANDOM
            cleanup_cpu() {{ {self._quark_force_delete()}; }}
            trap cleanup_cpu EXIT INT TERM
            {self._quark_timed_create_start()}
            """
        ).strip()

        exec_hot_body = textwrap.dedent(
            f"""
            t0=$(date +%s%N)
            timeout {exec_timeout} {exec_line} >/dev/null
            t1=$(date +%s%N)
            echo "METRIC exec_in_running_ms SAMPLE $__batch_i $(( (t1 - t0) / 1000000 ))"
            """
        ).strip()

        exec_hot_preamble = textwrap.dedent(
            f"""
            ID=keska-exec-hot-$RANDOM
            cleanup_hot() {{ {self._quark_force_delete()}; }}
            trap cleanup_hot EXIT INT TERM
            {self._quark_timed_create_start()}
            """
        ).strip()

        load_teardown = textwrap.dedent(
            """
            cleanup_load
            trap - EXIT INT TERM
            """
        ).strip()

        cpu_teardown = textwrap.dedent(
            """
            cleanup_cpu
            trap - EXIT INT TERM
            """
        ).strip()

        exec_hot_teardown = textwrap.dedent(
            """
            cleanup_hot
            trap - EXIT INT TERM
            """
        ).strip()

        return "\n".join(
            [
                "set -euo pipefail",
                self._bundle_batch_preamble(image),
                batch_warmup,
                remote_batch_loop(n, vm_boot_body),
                remote_batch_loop(n, tti_body),
                load_setup,
                remote_batch_loop(n, tti_load_body),
                load_teardown,
                remote_batch_loop(n, mem_body),
                remote_batch_loop(n, pause_body),
                cpu_preamble,
                remote_batch_loop(n, cpu_loop_body),
                cpu_teardown,
                exec_hot_preamble,
                remote_batch_loop(n, exec_hot_body),
                exec_hot_teardown,
            ]
        )

    def run_light_suite_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        exec_cmd: str = "/bin/echo ok",
        cpu_loop_cmd: str | None = None,
        idle_cmd: str = "/bin/sleep 600",
    ) -> dict[str, list[float]]:
        del idle_cmd
        if self.exec_mode == "docker":
            raise NotImplementedError("run_light_suite_batch requires direct OCI exec_mode")
        if is_postgres_workload(image):
            raise NotImplementedError("run_light_suite_batch for postgres not supported")
        inner = normalize_cpu_loop_inner(cpu_loop_cmd)
        script = self._light_suite_batch_script(
            n, image=image, exec_cmd=exec_cmd, cpu_loop_inner=inner
        )
        timeout = n * 60 + 300
        r = self.remote.sh(script, timeout=timeout, check=True)

        grouped = parse_metric_samples(r.stdout)
        expected = {
            "vm_boot_ms": n,
            "tti_ms": n,
            "tti_under_load_ms": n,
            "memory_idle_rss_mb": n,
            "pause_ms": n,
            "resume_ms": n,
            "memory_while_paused_rss_mb": n,
            "cpu_loop_ms": n,
            "exec_in_running_ms": n,
        }
        for metric, count in expected.items():
            got = grouped.get(metric, [])
            if len(got) != count:
                seen: set[int] = set()
                for line in r.stdout.splitlines():
                    parts = line.split()
                    if (
                        len(parts) >= 5
                        and parts[0] == "METRIC"
                        and parts[1] == metric
                        and parts[2] == "SAMPLE"
                    ):
                        seen.add(int(parts[3]))
                missing = [i for i in range(1, count + 1) if i not in seen]
                raise ValueError(
                    f"light suite expected {count} {metric} samples, "
                    f"got {len(got)} (missing indices: {missing}) from:\n{r.stdout!r}"
                )
        return grouped

    def tti_under_load_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        load: int = 4,
        exec_cmd: str = "/bin/echo ok",
    ) -> list[float]:
        q = self._quark_cmd
        iter_body = textwrap.dedent(
            f"""
            ID=keska-tti-$RANDOM
            t0=$(date +%s%N)
            {self._quark_timed_create_start()}
            timeout 60 {q(f'exec --user 0:0 "$ID" -- {exec_cmd}')}
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            {self._quark_force_delete()}
            """
        ).strip()
        load_setup = textwrap.dedent(
            f"""
            LOAD_IDS=""
            cleanup_load() {{
              for id in $LOAD_IDS; do
                {self._quark_force_delete('"$id"')}
              done
            }}
            for i in $(seq 1 {load}); do
              lid=keska-load-$RANDOM-$i
              LOAD_IDS="$LOAD_IDS $lid"
              timeout 30 {q('create "$lid" -b "$BUNDLE"')} >/dev/null
              timeout 60 {q('start "$lid"')} >/dev/null
            done
            sleep {MEMORY_RSS_SETTLE_SECS}
            trap cleanup_load EXIT INT TERM
            """
        ).strip()
        script = remote_batch_script(
            preamble=f"{self._bundle_batch_preamble(image)}\n{load_setup}",
            n=n,
            body=iter_body,
        )
        r = self.remote.sh(script, timeout=n * 120 + 120, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def cpu_loop_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        exec_cmd: str | None = None,
    ) -> list[float]:
        inner = normalize_cpu_loop_inner(exec_cmd)
        q = self._quark_cmd
        pg_wait = self._postgres_startup_sleep() if is_postgres_workload(image) else ""
        cleanup_rm = 'sudo rm -rf "$BUNDLE" 2>/dev/null || true' if is_postgres_workload(image) else ""
        exec_line = self._quark_exec_sh_inner('"$ID"', inner)
        iter_body = textwrap.dedent(
            f"""
            t0=$(date +%s%N)
            timeout 120 {exec_line} >/dev/null
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            """
        ).strip()
        preamble = textwrap.dedent(
            f"""
            {self._bundle_batch_preamble(image)}
            ID=keska-cpu-$RANDOM
            cleanup() {{
              {self._quark_force_delete()}
              {cleanup_rm}
            }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            {pg_wait}
            """
        ).strip()
        script = remote_batch_script(preamble=preamble, n=n, body=iter_body)
        r = self.remote.sh(script, timeout=n * 180 + 60, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def exec_hot_once(
        self,
        *,
        image: str = "busybox",
        exec_cmd: str = "/bin/echo ok",
    ) -> float:
        return self.exec_hot_batch(1, image=image, exec_cmd=exec_cmd)[0]

    def exec_hot_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        exec_cmd: str = "/bin/echo ok",
    ) -> list[float]:
        if self.exec_mode == "docker":
            raise NotImplementedError("exec_hot_batch requires direct OCI exec_mode")
        q = self._quark_cmd
        pg_wait = self._postgres_startup_sleep() if is_postgres_workload(image) else ""
        cleanup_rm = 'sudo rm -rf "$BUNDLE" 2>/dev/null || true' if is_postgres_workload(image) else ""
        exec_line = q(f'exec --user 0:0 "$ID" -- {exec_cmd}')
        iter_body = textwrap.dedent(
            f"""
            t0=$(date +%s%N)
            {exec_line} >/dev/null
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            """
        ).strip()
        preamble = textwrap.dedent(
            f"""
            {self._bundle_batch_preamble(image)}
            ID=keska-exec-hot-$RANDOM
            cleanup() {{
              {self._quark_force_delete()}
              {cleanup_rm}
            }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            {pg_wait}
            """
        ).strip()
        script = remote_batch_script(preamble=preamble, n=n, body=iter_body)
        r = self.remote.sh(script, timeout=n * 30 + 120, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def micro_bench_once(
        self,
        *,
        metric: str,
        image: str = "python:3.12-slim",
    ) -> float:
        return self.micro_bench_batch(1, metric=metric, image=image)[0]

    def micro_bench_batch(
        self,
        n: int,
        *,
        metric: str,
        image: str = "python:3.12-slim",
    ) -> list[float]:
        if self.exec_mode == "docker":
            raise NotImplementedError("micro_bench_batch requires direct OCI exec_mode")
        code = micro_bench_script(metric)
        py = python_exec_cmd(code)
        q = self._quark_cmd
        bundle = self._bundle_path(image)
        iter_body = textwrap.dedent(
            f"""
            val=$({q('exec --user 0:0 "$ID" --')} {py})
            echo "$val" | tail -1
            """
        ).strip()
        preamble = textwrap.dedent(
            f"""
            BUNDLE={shlex.quote(bundle)}
            {self._refresh_rootfs(image, bundle_var="BUNDLE")}
            ID=keska-micro-$RANDOM
            cleanup() {{ {self._quark_force_delete()}; }}
            trap cleanup EXIT INT TERM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            """
        ).strip()
        script = remote_batch_script(preamble=preamble, n=n, body=iter_body)
        per = 180 if metric == "mmap_anon_fault_ms" else 60
        r = self.remote.sh(script, timeout=n * per + 120, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def io_fs_batch(self, n: int, *, image: str = "busybox") -> tuple[list[float], list[float]]:
        q = self._quark_cmd
        prep = quark_io_bench_bundle_preamble(self.config, image)
        cleanup_trap = quark_io_bench_cleanup_trap(
            quark_delete_cmd=self._quark_force_delete(),
        )
        io_exec = q(f'exec --user 0:0 "$ID" -- sh -c {shlex.quote(dd_io_bench_sh())}')
        body = textwrap.dedent(
            f"""
            set -euo pipefail
            {prep}
            {cleanup_trap}
            ID=keska-io-$RANDOM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            out=$({io_exec})
            cleanup_io_bench
            trap - EXIT INT TERM
            echo "$out"
            """
        ).strip()
        script = remote_batch_loop(n, body)
        r = self.remote.sh(script, timeout=n * 180 + 60, check=True)
        writes: list[float] = []
        reads: list[float] = []
        for line in r.stdout.splitlines():
            if line.startswith("WRITE "):
                writes.append(parse_dd_mib_s(line))
            elif line.startswith("READ "):
                reads.append(parse_dd_mib_s(line))
        if len(writes) != n or len(reads) != n:
            raise ValueError(
                f"io_fs_batch expected {n} write/read pairs, got {len(writes)}/{len(reads)}"
            )
        return writes, reads

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
        prep = quark_io_bench_bundle_preamble(self.config, image)
        cleanup = quark_io_bench_cleanup_trap(
            quark_delete_cmd=self._quark_force_delete(),
        )
        io_exec = q(
            f'exec --user 0:0 "$ID" -- sh -c {shlex.quote(dd_io_bench_sh())}'
        )
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            {prep}
            {cleanup}
            ID=keska-io-$RANDOM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            out=$({io_exec})
            cleanup_io_bench
            trap - EXIT INT TERM
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

    def io_fs_concurrent_read_once(self, *, image: str = "busybox") -> float:
        q = self._quark_cmd
        prep = quark_io_bench_bundle_preamble(self.config, image)
        cleanup = quark_io_bench_cleanup_trap(
            quark_delete_cmd=self._quark_force_delete(),
        )
        bench = concurrent_read_bench_sh()
        io_exec = q(
            f'exec --user 0:0 "$ID" -- sh -c {shlex.quote(bench)}'
        )
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            {prep}
            {cleanup}
            ID=keska-io-c-$RANDOM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            t0=$(date +%s%N)
            out=$({io_exec})
            t1=$(date +%s%N)
            cleanup_io_bench
            trap - EXIT INT TERM
            echo "$out"
            echo ELAPSED_NS=$((t1 - t0))
            """
        ).strip()
        r = self.remote.sh(script, timeout=180, check=True)
        if "CONCURRENT_READ_OK" not in r.stdout:
            raise RuntimeError(f"concurrent read bench failed: {r.stdout!r}")
        elapsed_ns = 0
        for line in r.stdout.splitlines():
            if line.startswith("ELAPSED_NS="):
                elapsed_ns = int(line.split("=", 1)[1])
        return concurrent_read_mib_s(
            total_bytes=concurrent_read_total_bytes(),
            elapsed_ns=elapsed_ns,
        )

    def file_io_integrity_once(self, *, image: str = "busybox") -> None:
        """Write/read 4 MiB on a host-backed bind mount and verify sha256."""
        q = self._quark_cmd
        prep = quark_io_bench_bundle_preamble(self.config, image)
        cleanup = quark_io_bench_cleanup_trap(
            quark_delete_cmd=self._quark_force_delete(),
        )
        bench_file = IO_BENCH_FILE
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            {prep}
            {cleanup}
            ID=keska-fio-$RANDOM
            {q('create "$ID" -b "$BUNDLE"')}
            {q('start "$ID"')}
            {q(
                'exec --user 0:0 "$ID" -- sh -c '
                f'"dd if=/dev/urandom of={shlex.quote(bench_file)} bs=1M count=4 conv=fsync >/dev/null 2>&1 && '
                f'w=$(sha256sum {shlex.quote(bench_file)} | awk \"{{print \\$1}}\") && '
                f'dd if={shlex.quote(bench_file)} of=/dev/null bs=1M >/dev/null 2>&1 && '
                f'r=$(sha256sum {shlex.quote(bench_file)} | awk \"{{print \\$1}}\") && '
                'test \\"$w\\" = \\"$r\\" && echo OK $w"'
            )}
            cleanup_io_bench
            trap - EXIT INT TERM
            """
        ).strip()
        r = self.remote.sh(script, timeout=120, check=True)
        if "OK " not in r.stdout:
            raise RuntimeError(f"file I/O integrity check failed: {r.stdout!r}")

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
            cleanup() {{ {self._quark_force_delete()}; }}
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
        probe = self.probe()
        if probe.get("tsot_ready") or probe.get("cri_ready"):
            return "crictl"
        return self.exec_mode

    def inet_tcp_connect_once_ms(self, *, image: str = "python:3.12-slim") -> float:
        probe = self.probe()
        if not probe.get("tsot_ready") and not probe.get("cri_ready"):
            raise RuntimeError(
                f"Quark network not ready (TSOT/CRI). Run: lab.quark.bench('network', setup=True)"
            )
        if self._network_exec_mode() == "crictl":
            return self._crictl_python_exec_sample(INET_CONNECT_PY, image=image)
        return self._python_exec_sample(INET_CONNECT_PY, image=image)

    def inet_download_mbps_once(self, *, image: str = "python:3.12-slim") -> float:
        probe = self.probe()
        if not probe.get("tsot_ready") and not probe.get("cri_ready"):
            raise RuntimeError(
                f"Quark network not ready (TSOT/CRI). Run: lab.quark.bench('network', setup=True)"
            )
        if self._network_exec_mode() == "crictl":
            return self._crictl_python_exec_sample(INET_DOWNLOAD_PY, image=image, timeout=120)
        return self._python_exec_sample(INET_DOWNLOAD_PY, image=image, timeout=120)

    def sandbox_iperf_mbps_once(self, *, image: str = "networkstatic/iperf3") -> float:
        probe = self.probe()
        if not probe.get("tsot_ready") and not probe.get("cri_ready"):
            raise RuntimeError(
                f"Quark network not ready (TSOT/CRI). Run: lab.quark.bench('network', setup=True)"
            )
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
