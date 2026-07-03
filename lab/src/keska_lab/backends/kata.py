"""Kata Containers backend — containerd ctr (no dockerd in hot path)."""

from __future__ import annotations

import shlex
import textwrap

from keska_lab.backends.base import SandboxBackend
from keska_lab.config import LabConfig
from keska_lab.harness.batch import remote_batch_loop, remote_batch_script
from keska_lab.harness.metrics import (
    MEMORY_RSS_SETTLE_SECS,
    parse_float_lines,
    parse_pause_resume_lines,
    rss_by_args_match_shell,
)
from keska_lab.harness.io_fs import (
    concurrent_read_bench_sh,
    concurrent_read_mib_s,
    concurrent_read_total_bytes,
    dd_io_bench_sh,
    io_bench_host_dir,
)
from keska_lab.setup.oci_bundle import postgres_data_template_dir
from keska_lab.harness.network import (
    INET_CONNECT_PY,
    INET_DOWNLOAD_PY,
    crictl_iperf_script,
    crictl_python_exec_script,
    parse_dd_mib_s,
    parse_iperf_mbps,
    python_exec_cmd,
)
from keska_lab.harness.postgres import (
    POSTGRES_STARTUP_SLEEP_SECS,
    kata_postgres_ctr_mounts_shell,
)
from keska_lab.harness.workload import (
    is_postgres_workload,
    micro_bench_script,
    normalize_cpu_loop_inner,
)
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

    def _pg_ready_wait_secs(self, timeout: int, *, exec_timeout: int = 8) -> int:
        overhead = 45
        per_iter = exec_timeout + 1
        budget = max(10, timeout - overhead)
        return min(60, budget // per_iter)

    def _kata_kill_rm(self, id_ref: str = '"$ID"') -> str:
        return (
            f"sudo -n ctr task kill -s SIGKILL {id_ref} >/dev/null 2>&1 || true\n"
            f"            sudo -n ctr containers rm {id_ref} >/dev/null 2>&1 || true"
        )

    def _detached_lifecycle_script(
        self,
        *,
        image: str,
        probe: str,
        idle_cmd: str = "/bin/sleep 3600",
        id_prefix: str = "keska-kata",
        exec_timeout: int = 60,
    ) -> str:
        ref = self._image_ref(image)
        env = self._ctr_env_flags(image)
        return textwrap.dedent(
            f"""
            set -euo pipefail
            ID={id_prefix}-$RANDOM
            cleanup() {{
              {self._kata_kill_rm()}
            }}
            trap cleanup EXIT INT TERM
            t0=$(date +%s%N)
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {shlex.quote(ref)} "$ID" {idle_cmd} >/dev/null
            timeout {exec_timeout} sudo -n ctr task exec --exec-id probe-$RANDOM "$ID" {probe} >/dev/null
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            cleanup
            trap - EXIT INT TERM
            """
        ).strip()

    def _postgres_data_prep_script(self) -> str:
        tmpl = postgres_data_template_dir(self.config)
        return textwrap.dedent(
            f"""
            TMPL={shlex.quote(tmpl)}
            sudo test -f "$TMPL/PG_VERSION" || {{
              echo "missing postgres data template at $TMPL (run db setup)" >&2
              exit 1
            }}
            DATA=/tmp/keska-kata-pgdata-$RANDOM
            mkdir -p "$DATA"
            if ! mountpoint -q "$DATA" 2>/dev/null; then
              sudo -n mount -t tmpfs -o size=512m,mode=1777 tmpfs "$DATA"
            fi
            sudo cp -a "$TMPL/." "$DATA/"
            sudo chown -R 70:70 "$DATA"
            sudo chmod 700 "$DATA"
            PGDATA_MOUNT="type=bind,src=$DATA,dst=/var/lib/postgresql/data,options=rbind:rw"
            {kata_postgres_ctr_mounts_shell()}
            trap 'sudo -n umount "$DATA" 2>/dev/null || true' EXIT INT TERM
            """
        ).strip()

    def _kata_postgres_mount_flags(self) -> str:
        return '--mount "$PGDATA_MOUNT" --mount "$PG_SHM_MOUNT" --mount "$PG_RUN_MOUNT"'

    def _postgres_tti_script(
        self,
        *,
        image: str,
        probe: str,
        idle_cmd: str,
        wait_secs: int = 600,
        exec_timeout: int = 8,
    ) -> str:
        ref = self._image_ref(image)
        env = self._ctr_env_flags(image)
        prep = self._postgres_data_prep_script()
        return textwrap.dedent(
            f"""
            set -euo pipefail
            {prep}
            ID=keska-kata-pg-tti-$RANDOM
            cleanup() {{
              {self._kata_kill_rm()}
            }}
            trap cleanup EXIT INT TERM
            t0=$(date +%s%N)
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {self._kata_postgres_mount_flags()} \\
              {shlex.quote(ref)} "$ID" {idle_cmd} >/dev/null
            ready=0
            for i in $(seq 1 {wait_secs}); do
              if timeout {exec_timeout} sudo -n ctr task exec --user 70:70 --exec-id chk-$RANDOM "$ID" {probe} >/dev/null 2>&1; then
                ready=1
                break
              fi
              sleep 1
            done
            test "$ready" = 1
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            cleanup
            trap - EXIT INT TERM
            """
        ).strip()

    def _vm_boot_iter_body(
        self,
        *,
        image: str,
        idle_cmd: str = "/bin/sleep 3600",
        id_prefix: str = "keska-kata-boot",
    ) -> str:
        ref = self._image_ref(image)
        env = self._ctr_env_flags(image)
        return textwrap.dedent(
            f"""
            ID={id_prefix}-$RANDOM
            cleanup() {{
              {self._kata_kill_rm()}
            }}
            trap cleanup EXIT INT TERM
            t0=$(date +%s%N)
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {shlex.quote(ref)} "$ID" {idle_cmd} >/dev/null
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            cleanup
            trap - EXIT INT TERM
            """
        ).strip()

    def _kata_task_pid_shell(self, id_ref: str = '"$ID"') -> str:
        return (
            f'__task_pid=$(sudo -n ctr task ls 2>/dev/null | awk -v id={id_ref} '
            f"'$1==id {{print $2; exit}}')"
        )

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
            exec_timeout = 8
            wait = self._pg_ready_wait_secs(timeout, exec_timeout=exec_timeout)
            script = self._postgres_tti_script(
                image=image,
                probe=exec_cmd,
                idle_cmd="docker-entrypoint.sh postgres",
                wait_secs=wait,
                exec_timeout=exec_timeout,
            )
            timeout = min(timeout, 45 + wait * (exec_timeout + 1) + 30)
        else:
            script = self._detached_lifecycle_script(
                image=image,
                probe=exec_cmd,
            )
        r = self.remote.sh(script, timeout=timeout, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def stress_once(self, *, image: str = "busybox", wave_index: int = 0) -> float:
        del wave_index
        r = self.remote.sh(
            self._detached_lifecycle_script(
                image=image,
                probe="/bin/true",
                id_prefix="keska-kata-stress",
            ),
            timeout=120,
            check=False,
        )
        if not r.ok:
            raise RuntimeError(r.stderr.strip() or "kata stress sample failed")
        return float(r.stdout.strip().splitlines()[-1])

    def _rss_for_sandbox(self, id_shell_var: str = "$ID") -> str:
        """Awk snippet: sum RSS (MB) for processes tied to a sandbox ID."""
        return rss_by_args_match_shell(id_shell_var)

    def memory_idle_once(self, *, image: str = "busybox", idle_cmd: str = "/bin/sleep 600") -> float:
        ref = self._image_ref(image)
        env = self._ctr_env_flags(image)
        prep = ""
        pg_mount_line = ""
        pg_wait = ""
        if is_postgres_workload(image):
            prep = self._postgres_data_prep_script()
            pg_mount_line = f"{self._kata_postgres_mount_flags()} \\"
            pg_wait = f"sleep {POSTGRES_STARTUP_SLEEP_SECS}"
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            {prep}
            ID=keska-kata-mem-$RANDOM
            cleanup() {{
              {self._kata_kill_rm()}
            }}
            trap cleanup EXIT INT TERM
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {pg_mount_line}  {shlex.quote(ref)} "$ID" {idle_cmd} >/dev/null
            {pg_wait}
            sleep 2
            {self._rss_for_sandbox("$ID")}
            cleanup
            trap - EXIT INT TERM
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
                {shlex.quote(ref)} "$lid" /bin/sleep 3600 >/dev/null
            done
            sleep 1
            ID=keska-kata-tti-$RANDOM
            t0=$(date +%s%N)
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()} \\
              {shlex.quote(ref)} "$ID" /bin/sleep 3600 >/dev/null
            timeout 60 sudo -n ctr task exec --exec-id probe-$RANDOM "$ID" {exec_cmd} >/dev/null
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            {self._kata_kill_rm()}
            cleanup
            """
        ).strip()
        r = self.remote.sh(script, timeout=360, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def _cpu_loop_iter_body(
        self,
        *,
        image: str,
        inner_sh: str,
        id_prefix: str = "keska-kata-cpu",
    ) -> str:
        ref = self._image_ref(image)
        return textwrap.dedent(
            f"""
            ID={id_prefix}-$RANDOM
            cleanup() {{
              {self._kata_kill_rm()}
            }}
            trap cleanup EXIT INT TERM
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()} \\
              {shlex.quote(ref)} "$ID" /bin/sleep 3600 >/dev/null
            t0=$(date +%s%N)
            timeout 120 sudo -n ctr task exec --exec-id cpu-$RANDOM "$ID" /bin/sh -c {shlex.quote(inner_sh)} >/dev/null
            t1=$(date +%s%N)
            cleanup
            trap - EXIT INT TERM
            echo $(( (t1 - t0) / 1000000 ))
            """
        ).strip()

    def _cpu_loop_body(
        self,
        *,
        image: str,
        exec_cmd: str,
        id_prefix: str = "keska-kata-cpu",
    ) -> str:
        inner = normalize_cpu_loop_inner(exec_cmd)
        return textwrap.dedent(
            f"""
            set -euo pipefail
            {self._cpu_loop_iter_body(image=image, inner_sh=inner, id_prefix=id_prefix)}
            """
        ).strip()

    def cpu_loop_once(
        self,
        *,
        image: str = "busybox",
        exec_cmd: str | None = None,
    ) -> float:
        inner = normalize_cpu_loop_inner(exec_cmd)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            {self._cpu_loop_iter_body(image=image, inner_sh=inner)}
            """
        ).strip()
        r = self.remote.sh(script, timeout=180, check=True)
        return float(r.stdout.strip().splitlines()[-1])

    def vm_boot_once(
        self,
        *,
        image: str = "busybox",
        idle_cmd: str = "/bin/sleep 600",
    ) -> float:
        return self.vm_boot_batch(1, image=image, idle_cmd=idle_cmd)[0]

    def vm_boot_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        idle_cmd: str = "/bin/sleep 600",
    ) -> list[float]:
        if is_postgres_workload(image):
            raise NotImplementedError("vm_boot_batch for postgres not supported")
        body = self._vm_boot_iter_body(image=image, idle_cmd=idle_cmd)
        iter_body = body.replace("set -euo pipefail\n", "", 1)
        script = remote_batch_script(preamble="", n=n, body=iter_body)
        r = self.remote.sh(script, timeout=n * 60 + 60, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def tti_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        exec_cmd: str = "/bin/echo ok",
        timeout: int = 300,
    ) -> list[float]:
        if is_postgres_workload(image, exec_cmd):
            raise NotImplementedError("tti_batch for postgres not supported")
        body = self._detached_lifecycle_script(image=image, probe=exec_cmd)
        iter_body = body.replace("set -euo pipefail\n", "", 1)
        script = remote_batch_script(preamble="", n=n, body=iter_body)
        r = self.remote.sh(script, timeout=n * min(timeout, 30) + 60, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def memory_idle_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        idle_cmd: str = "/bin/sleep 600",
    ) -> list[float]:
        ref = self._image_ref(image)
        env = self._ctr_env_flags(image)
        prep = ""
        pg_mount_line = ""
        pg_wait = ""
        if is_postgres_workload(image):
            prep = self._postgres_data_prep_script()
            pg_mount_line = f"{self._kata_postgres_mount_flags()} \\"
            pg_wait = f"sleep {POSTGRES_STARTUP_SLEEP_SECS}"
        iter_body = textwrap.dedent(
            f"""
            ID=keska-kata-mem-$RANDOM
            cleanup() {{
              {self._kata_kill_rm()}
            }}
            trap cleanup EXIT INT TERM
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {pg_mount_line}  {shlex.quote(ref)} "$ID" {idle_cmd} >/dev/null
            {pg_wait}
            sleep {MEMORY_RSS_SETTLE_SECS}
            {self._rss_for_sandbox("$ID")}
            cleanup
            trap - EXIT INT TERM
            echo "${{rss:-0}}"
            """
        ).strip()
        script = remote_batch_script(preamble=prep, n=n, body=iter_body)
        r = self.remote.sh(script, timeout=n * 60 + 60, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def pause_resume_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        idle_cmd: str = "/bin/sleep 600",
    ) -> list[tuple[float, float, float]]:
        ref = self._image_ref(image)
        iter_body = textwrap.dedent(
            f"""
            ID=keska-kata-pause-$RANDOM
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()} \\
              {shlex.quote(ref)} "$ID" {idle_cmd} >/dev/null
            sleep {MEMORY_RSS_SETTLE_SECS}
            t0=$(date +%s%N)
            sudo -n ctr task pause "$ID" >/dev/null 2>&1
            t1=$(date +%s%N)
            sleep {MEMORY_RSS_SETTLE_SECS}
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
        script = remote_batch_script(preamble="", n=n, body=iter_body)
        r = self.remote.sh(script, timeout=n * 60 + 60, check=True)
        return parse_pause_resume_lines(r.stdout, expect=n)

    def tti_under_load_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        load: int = 4,
        exec_cmd: str = "/bin/echo ok",
    ) -> list[float]:
        ref = self._image_ref(image)
        iter_body = textwrap.dedent(
            f"""
            ID=keska-kata-tti-$RANDOM
            t0=$(date +%s%N)
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()} \\
              {shlex.quote(ref)} "$ID" /bin/sleep 3600 >/dev/null
            timeout 60 sudo -n ctr task exec --exec-id probe-$RANDOM "$ID" {exec_cmd} >/dev/null
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            {self._kata_kill_rm()}
            """
        ).strip()
        load_setup = textwrap.dedent(
            f"""
            LOAD_IDS=""
            cleanup_load() {{
              for id in $LOAD_IDS; do
                sudo -n ctr task kill -s SIGKILL "$id" >/dev/null 2>&1 || true
                sudo -n ctr containers rm "$id" >/dev/null 2>&1 || true
              done
            }}
            for i in $(seq 1 {load}); do
              lid=keska-kata-load-$RANDOM-$i
              LOAD_IDS="$LOAD_IDS $lid"
              sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()} \\
                {shlex.quote(ref)} "$lid" /bin/sleep 3600 >/dev/null
            done
            sleep 1
            trap cleanup_load EXIT INT TERM
            """
        ).strip()
        script = remote_batch_script(
            preamble=load_setup,
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
        script = remote_batch_script(
            preamble="",
            n=n,
            body=self._cpu_loop_iter_body(image=image, inner_sh=inner),
        )
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
        ref = self._image_ref(image)
        env = self._ctr_env_flags(image)
        prep = ""
        pg_mount_line = ""
        pg_wait = ""
        if is_postgres_workload(image):
            prep = self._postgres_data_prep_script()
            pg_mount_line = f"{self._kata_postgres_mount_flags()} \\"
            pg_wait = f"sleep {POSTGRES_STARTUP_SLEEP_SECS}"
        iter_body = textwrap.dedent(
            f"""
            t0=$(date +%s%N)
            sudo -n ctr task exec --exec-id hot-$RANDOM "$ID" {exec_cmd} >/dev/null
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            """
        ).strip()
        preamble = textwrap.dedent(
            f"""
            {prep}
            ID=keska-kata-exec-hot-$RANDOM
            cleanup() {{
              {self._kata_kill_rm()}
            }}
            trap cleanup EXIT INT TERM
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {pg_mount_line}  {shlex.quote(ref)} "$ID" /bin/sleep 3600 >/dev/null
            {pg_wait}
            """
        ).strip()
        script = remote_batch_script(preamble=preamble, n=n, body=iter_body)
        r = self.remote.sh(script, timeout=n * 30 + 120, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def exec_nsenter_once(
        self,
        *,
        image: str = "busybox",
        exec_cmd: str = "/bin/echo ok",
    ) -> float:
        return self.exec_nsenter_batch(1, image=image, exec_cmd=exec_cmd)[0]

    def exec_nsenter_batch(
        self,
        n: int,
        *,
        image: str = "busybox",
        exec_cmd: str = "/bin/echo ok",
    ) -> list[float]:
        ref = self._image_ref(image)
        env = self._ctr_env_flags(image)
        exec_timeout = 15
        iter_body = textwrap.dedent(
            f"""
            {self._kata_task_pid_shell()}
            if [ -z "${{__task_pid:-}}" ]; then
              echo "ERR missing task pid for $ID" >&2
              exit 1
            fi
            t0=$(date +%s%N)
            timeout {exec_timeout} sudo -n nsenter --target "$__task_pid" --mount --pid -- {exec_cmd} >/dev/null
            t1=$(date +%s%N)
            echo $(( (t1 - t0) / 1000000 ))
            """
        ).strip()
        preamble = textwrap.dedent(
            f"""
            ID=keska-kata-nsenter-$RANDOM
            cleanup() {{
              {self._kata_kill_rm()}
            }}
            trap cleanup EXIT INT TERM
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {shlex.quote(ref)} "$ID" /bin/sleep 3600 >/dev/null
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
        ref = self._image_ref(image)
        env = self._ctr_env_flags(image)
        code = micro_bench_script(metric)
        py = python_exec_cmd(code)
        iter_body = textwrap.dedent(
            f"""
            val=$(sudo -n ctr task exec --exec-id micro-$RANDOM "$ID" {py})
            echo "$val" | tail -1
            """
        ).strip()
        preamble = textwrap.dedent(
            f"""
            ID=keska-kata-micro-$RANDOM
            cleanup() {{
              {self._kata_kill_rm()}
            }}
            trap cleanup EXIT INT TERM
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {shlex.quote(ref)} "$ID" python3 -c "import time; time.sleep(3600)" >/dev/null
            """
        ).strip()
        script = remote_batch_script(preamble=preamble, n=n, body=iter_body)
        per = 180 if metric == "mmap_anon_fault_ms" else 60
        r = self.remote.sh(script, timeout=n * per + 120, check=True)
        return parse_float_lines(r.stdout, expect=n)

    def io_fs_batch(self, n: int, *, image: str = "busybox") -> tuple[list[float], list[float]]:
        body = self._kata_io_bench_script(
            image=image,
            guest_cmd=dd_io_bench_sh(),
            id_prefix="keska-kata-io",
        )
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

    def _kata_io_bench_script(
        self,
        *,
        image: str,
        guest_cmd: str,
        id_prefix: str,
        timed: bool = False,
    ) -> str:
        """Detached ctr sandbox + task exec on /bench (teardown not timed)."""
        ref = self._image_ref(image)
        host_root = io_bench_host_dir(self.config)
        timing_start = "t0=$(date +%s%N)\n            " if timed else ""
        timing_end = (
            "t1=$(date +%s%N)\n            echo ELAPSED_NS=$((t1 - t0))\n            "
            if timed
            else ""
        )
        cmd_q = shlex.quote(guest_cmd)
        return textwrap.dedent(
            f"""
            set -euo pipefail
            HOST_BENCH={shlex.quote(host_root)}/run-$RANDOM
            sudo -n mkdir -p {shlex.quote(host_root)} "$HOST_BENCH"
            sudo -n chmod 755 "$HOST_BENCH"
            cleanup() {{
              {self._kata_kill_rm()}
              sudo -n rm -rf "$HOST_BENCH" 2>/dev/null || true
            }}
            trap cleanup EXIT INT TERM
            ID={id_prefix}-$RANDOM
            MOUNT="type=bind,src=$HOST_BENCH,dst=/bench,options=rbind:rw"
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()} \\
              --mount "$MOUNT" \\
              {shlex.quote(ref)} "$ID" /bin/sleep 3600 >/dev/null
            {timing_start}out=$(sudo -n ctr task exec --exec-id io-$RANDOM "$ID" sh -c {cmd_q} 2>&1)
            {timing_end}echo "$out"
            cleanup
            trap - EXIT INT TERM
            """
        ).strip()

    def io_fs_once(self, *, image: str = "busybox") -> tuple[float, float]:
        script = self._kata_io_bench_script(
            image=image,
            guest_cmd=dd_io_bench_sh(),
            id_prefix="keska-kata-io",
        )
        r = self.remote.sh(script, timeout=180, check=True)
        write_mib_s = read_mib_s = 0.0
        for line in r.stdout.splitlines():
            if line.startswith("WRITE "):
                write_mib_s = parse_dd_mib_s(line)
            elif line.startswith("READ "):
                read_mib_s = parse_dd_mib_s(line)
        return write_mib_s, read_mib_s

    def io_fs_concurrent_read_once(self, *, image: str = "busybox") -> float:
        script = self._kata_io_bench_script(
            image=image,
            guest_cmd=concurrent_read_bench_sh(),
            id_prefix="keska-kata-io-c",
            timed=True,
        )
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
        prep = self._postgres_data_prep_script()
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            {prep}
            ID=keska-kata-pg-$RANDOM
            cleanup() {{
              {self._kata_kill_rm()}
            }}
            trap cleanup EXIT INT TERM
            sudo -n ctr run -d --runtime {shlex.quote(self.ctr_runtime)}{self._ctr_snapshotter_flag()}{env} \\
              {self._kata_postgres_mount_flags()} \\
              {shlex.quote(ref)} "$ID" {idle_cmd} >/dev/null
            sleep {POSTGRES_STARTUP_SLEEP_SECS}
            out=$(sudo -n ctr task exec --user 70:70 --exec-id pgb-$RANDOM "$ID" pgbench -c1 -T5 -U postgres 2>&1 || true)
            cleanup
            trap - EXIT INT TERM
            echo "$out"
            """
        ).strip()
        r = self.remote.sh(script, timeout=300, check=True)
        return parse_pgbench_tps(r.stdout)
