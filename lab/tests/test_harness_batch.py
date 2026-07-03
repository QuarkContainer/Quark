"""Harness batch script generation and parsing."""

from __future__ import annotations

import textwrap

from keska_lab.harness.batch import remote_batch_loop, remote_batch_script
from keska_lab.harness.metrics import (
    parse_float_lines,
    parse_metric_samples,
    parse_pause_resume_lines,
    quark_rss_for_sandbox_shell,
)
from keska_lab.harness.stats import MetricStats
from keska_lab.harness.suites import resolve_suite
from keska_lab.harness.workload import micro_bench_script, normalize_cpu_loop_inner
from keska_lab.setup.quark_cleanup import cleanup_script


def test_normalize_cpu_loop_inner_legacy():
    legacy = '/bin/sh -c "i=0; while [ $i -lt 2000000 ]; do i=$((i+1)); done; echo OK"'
    inner = normalize_cpu_loop_inner(legacy)
    assert "$i" in inner
    assert not inner.startswith("/bin/sh")


def test_quark_cpu_loop_exec_uses_shlex_quote():
    import shlex

    inner = normalize_cpu_loop_inner(None)
    quoted = shlex.quote(inner)
    line = f"exec --user 0:0 \"$ID\" -- /bin/sh -c {quoted}"
    assert line.startswith("exec --user 0:0")
    assert quoted[0] in "'\""
    assert "$i" in quoted


def test_cleanup_script_preserves_awk_braces():
    script = cleanup_script(io_bench_dir="/var/lib/keska-lab/io-bench")
    assert "awk 'NR>1 {print $1}'" in script
    assert "/var/lib/keska-lab/io-bench/run-*" in script


def test_remote_batch_loop_contains_seq():
    body = "echo 42"
    script = remote_batch_loop(5, body)
    assert "seq 1 5" in script
    assert "echo 42" in script
    assert "set +e" in script
    assert "set -euo pipefail" in script


def test_parse_float_lines_sample_format():
    stdout = "\n".join(f"SAMPLE {i} {v}" for i, v in enumerate([10.0, 20.5, 30.0], 1))
    assert parse_float_lines(stdout, expect=3) == [10.0, 20.5, 30.0]


def test_parse_float_lines_plain():
    stdout = "55\n56\n54\n"
    assert parse_float_lines(stdout, expect=3) == [55.0, 56.0, 54.0]


def test_parse_pause_resume_lines():
    stdout = "12 8 21.5\n11 9 22.0\n"
    triples = parse_pause_resume_lines(stdout, expect=2)
    assert triples == [(12.0, 8.0, 21.5), (11.0, 9.0, 22.0)]


def test_quark_rss_script_uses_list_and_id():
    qlist = "sudo -n '/usr/local/bin/quark' list"
    snippet = quark_rss_for_sandbox_shell(qlist, "$ID")
    assert qlist in snippet
    assert f"$({qlist} 2>/dev/null" in snippet
    assert "ID_VAL=$ID" in snippet
    # Must not quote the whole sudo/quark invocation as one command name.
    assert f"$('{qlist}'" not in snippet


def test_pause_batch_script_shape():
    from unittest.mock import MagicMock

    from keska_lab.backends.quark import QuarkBackend
    from keska_lab.config import LabConfig
    from keska_lab.harness.batch import remote_batch_script

    backend = QuarkBackend(MagicMock(config=LabConfig.from_env()))
    body = "printf '%s %s %s\\n' 1 2 3"
    script = remote_batch_script(
        preamble=backend._bundle_batch_preamble("busybox"),
        n=2,
        body=body,
    )
    assert "seq 1 2" in script
    assert "printf" in script


def test_micro_bench_script_getpid():
    script = micro_bench_script("getpid_ns")
    assert "getpid()" in script
    assert "100_000" in script


def test_micro_suite_cases():
    cases = resolve_suite("micro")
    names = {c.name for c in cases}
    assert "getpid_ns" in names
    assert "mmap_anon_fault_ms" in names
    assert "pipe_throughput_mib_s" in names


def test_parse_metric_samples_grouped():
    stdout = "\n".join(
        [
            "METRIC pause_ms SAMPLE 1 12",
            "METRIC resume_ms SAMPLE 1 8",
            "METRIC memory_while_paused_rss_mb SAMPLE 1 21.5",
        ]
    )
    grouped = parse_metric_samples(stdout)
    assert grouped["pause_ms"] == [12.0]
    assert grouped["resume_ms"] == [8.0]
    assert grouped["memory_while_paused_rss_mb"] == [21.5]


def test_quark_force_delete_uses_runtime_id():
    from unittest.mock import MagicMock

    from keska_lab.backends.quark import QuarkBackend
    from keska_lab.config import LabConfig

    backend = QuarkBackend(MagicMock(config=LabConfig.from_env()))
    snippet = backend._quark_force_delete()
    assert '"/run/qvisor/$ID/meta.json"' in snippet
    assert '"/run/qvisor/$ID"' in snippet
    assert '"/run/qvisor/ID"' not in snippet

    load_snippet = backend._quark_force_delete('"$id"')
    assert '"/run/qvisor/$id/meta.json"' in load_snippet
    assert '"/run/qvisor/$id"' in load_snippet


def test_force_delete_reads_meta_json_not_quark_list():
    from unittest.mock import MagicMock

    from keska_lab.backends.quark import QuarkBackend
    from keska_lab.config import LabConfig

    backend = QuarkBackend(MagicMock(config=LabConfig.from_env()))
    snippet = backend._quark_force_delete()
    assert "quark list" not in snippet
    assert "meta.json" in snippet
    assert "python3 -c" in snippet
    assert "Sandbox" in snippet
    assert "Pid" in snippet


def test_force_delete_load_id_variant():
    from unittest.mock import MagicMock

    from keska_lab.backends.quark import QuarkBackend
    from keska_lab.config import LabConfig

    backend = QuarkBackend(MagicMock(config=LabConfig.from_env()))
    snippet = backend._quark_force_delete('"$id"')
    assert '"/run/qvisor/$id/meta.json"' in snippet
    assert "quark list" not in snippet


def test_light_suite_script_no_per_iter_wildcard_rm():
    from unittest.mock import MagicMock

    from keska_lab.backends.quark import QuarkBackend
    from keska_lab.config import LabConfig

    remote = MagicMock()
    remote.config = LabConfig.from_env()
    backend = QuarkBackend(remote, profile="release", exec_mode="direct")
    script = backend._light_suite_batch_script(
        3, image="busybox", exec_cmd="/bin/echo ok", cpu_loop_inner="echo OK"
    )
    assert script.index("cp -a") < script.index("seq 1 3")
    assert script.count("/run/qvisor/keska-*") == 1
    assert "METRIC tti_ms SAMPLE" in script
    assert "METRIC exec_in_running_ms SAMPLE" in script
    assert script.index("cleanup_cpu") < script.rindex("METRIC cpu_loop_ms SAMPLE")
    assert "keska-warmup-" in script
    assert script.index("keska-warmup-") < script.index("METRIC vm_boot_ms SAMPLE")
    assert script.index("METRIC vm_boot_ms SAMPLE") < script.index("METRIC tti_ms SAMPLE")


def test_quark_vm_boot_batch_script_shape():
    from unittest.mock import MagicMock

    from keska_lab.backends.quark import QuarkBackend
    from keska_lab.config import LabConfig

    backend = QuarkBackend(MagicMock(config=LabConfig.from_env()))
    script = backend._direct_vm_boot_iter_body()
    assert "quark create" in script
    assert "quark start" in script
    assert "quark exec" not in script
    assert "quark delete" in script or "delete --force" in script


def test_kata_exec_nsenter_script_shape():
    from unittest.mock import MagicMock

    from keska_lab.backends.kata import KataBackend
    from keska_lab.config import LabConfig
    from keska_lab.harness.batch import remote_batch_script

    backend = KataBackend(MagicMock(config=LabConfig.from_env()))
    body = (
        backend._kata_task_pid_shell()
        + '\nnsenter --target "$__task_pid" --mount --pid -- /bin/echo ok'
    )
    script = remote_batch_script(
        preamble="ID=keska-kata-nsenter-$RANDOM",
        n=2,
        body=body,
    )
    assert "ctr task ls" in script
    assert "nsenter --target" in script
    assert "ctr task exec" not in script


def test_light_suite_includes_vm_boot_case():
    from keska_lab.harness.suites import resolve_suite

    names = [c.name for c in resolve_suite("light")]
    assert names[0] == "vm_boot_ms"
    assert "exec_nsenter_ms" not in names


def test_micro_suite_includes_exec_nsenter():
    from keska_lab.harness.suites import resolve_suite

    names = [c.name for c in resolve_suite("micro")]
    assert "exec_nsenter_ms" in names


def test_light_suite_exec_hot_uses_timeout():
    from unittest.mock import MagicMock

    from keska_lab.backends.quark import QuarkBackend
    from keska_lab.config import LabConfig

    backend = QuarkBackend(MagicMock(config=LabConfig.from_env()))
    script = backend._light_suite_batch_script(
        3, image="busybox", exec_cmd="/bin/echo ok", cpu_loop_inner="echo OK"
    )
    assert "timeout 15 sudo" in script.split("METRIC exec_in_running_ms")[0].split("exec-hot")[-1]
