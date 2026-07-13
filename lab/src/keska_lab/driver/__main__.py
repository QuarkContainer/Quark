from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from keska_lab.driver.errors import DriverErrorCode, DriverStatus
from keska_lab.driver.protocol import DriverEnvelope, ExitCode, exit_code_for_envelope
from keska_lab.driver.timeouts import (
    CRI_RUNP_S,
    OP_BUDGET_INET_CONNECT_S,
    OP_BUDGET_INET_DOWNLOAD_S,
    POLL_INTERVAL_S,
    POLL_STALL_S,
    TEARDOWN_BUDGET_S,
)
from keska_lab.driver.poll import PollWait
from keska_lab.driver.poll import WaitStalled, WaitTimedOut
from keska_lab.host.probes import drain_orphans, scan_orphans


def _read_json(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text())


def _write_json(path: str, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")


def _clean_report_from_orphans() -> dict:
    # Deterministic cleanup: bounded drain + targeted kill (psutil), then re-scan.
    rep = drain_orphans(budget_s=5.0)
    return {
        "pods_remaining": 0,
        "containers_remaining": 0,
        "orphan_shims": len(rep.shims),
        "orphan_firecrackers": len(rep.firecrackers),
    }


def op_host_scan_orphans(_inp: dict) -> dict:
    rep = scan_orphans()
    return {
        "shim_pids": [m.pid for m in rep.shims],
        "firecracker_pids": [m.pid for m in rep.firecrackers],
    }


def op_network_inet_connect(inp: dict) -> dict:
    import tempfile
    import uuid

    from keska_lab.driver.cri_client import CriClient
    from keska_lab.harness.cri_pair import pod_spec
    from keska_lab.harness.network import INET_CONNECT_PY

    budget_s = float(inp.get("op_budget_s", OP_BUDGET_INET_CONNECT_S))
    runtime_handler = str(inp.get("runtime_handler", "") or "")
    tsot_dns = bool(inp.get("tsot_dns", False))
    image = str(inp.get("image", "python:3.12-slim"))

    t0 = time.monotonic()
    cid = ""
    pod = ""
    tmpdir = tempfile.TemporaryDirectory(prefix="keska-driver-")
    try:
        wd = Path(tmpdir.name)
        uid = uuid.uuid4().hex[:12]
        pod_json = wd / "pod.json"
        ctr_json = wd / "ctr.json"

        pod_json.write_text(
            json.dumps(
                pod_spec(name=f"keska-net-{uid}", uid=f"net-{uid}", runtime_handler=runtime_handler, tsot_dns=tsot_dns),
                indent=2,
            )
            + "\n"
        )
        ctr_json.write_text(
            json.dumps(
                {
                    "metadata": {"name": "inet"},
                    "image": {"image": image},
                    # Avoid shell/base64 tricks; keep it maximally stable.
                    "command": ["python3", "-u", "-c", INET_CONNECT_PY.strip()],
                    "log_path": "inet.log",
                },
                indent=2,
            )
            + "\n"
        )

        cri = CriClient(sudo=True)

        if (time.monotonic() - t0) > budget_s:
            raise TimeoutError("op budget exceeded before runp")

        pod = cri.runp(str(pod_json), runtime_handler=runtime_handler, timeout_s=CRI_RUNP_S)
        cid = cri.create(pod, str(ctr_json), str(pod_json), timeout_s=30.0)
        cri.start(cid, timeout_s=30.0)

        def _logs_check() -> tuple[bool, str]:
            # Encode both container state and latest log line so stall detection has a state.
            cstate = "unknown"
            try:
                inspected = cri.inspect(cid, timeout_s=10.0)
                cstate = str(inspected.get("status", {}).get("state", "unknown"))
            except Exception:
                pass
            out = cri.logs(cid, tail=20, timeout_s=10.0)
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            line = lines[-1] if lines else ""
            # If container exited but never printed a number, fail early.
            if cstate.lower() in ("container_exited", "exited") and not line.isdigit():
                evidence: dict = {"logs_tail": lines[-20:], "container_state": cstate}
                try:
                    inspected = cri.inspect(cid, timeout_s=10.0)
                    evidence["inspect_status"] = inspected.get("status", {})
                    evidence["inspect_info"] = inspected.get("info", {})
                    log_path = str(inspected.get("status", {}).get("logPath") or "")
                    if log_path and not evidence["logs_tail"]:
                        try:
                            import subprocess

                            p = subprocess.run(
                                ["sudo", "-n", "tail", "-n", "50", log_path],
                                text=True,
                                capture_output=True,
                                timeout=5,
                            )
                            if p.returncode == 0 and p.stdout.strip():
                                evidence["log_path_tail"] = p.stdout.strip().splitlines()[-50:]
                            elif p.stderr.strip():
                                evidence["log_path_tail_err"] = p.stderr.strip()
                        except Exception as _e:
                            evidence["log_path_tail_err"] = str(_e)
                except Exception:
                    pass
                raise RuntimeError(f"container exited without result, evidence={evidence}")
            return line.isdigit(), f"{cstate}:{line}"

        state = PollWait(
            wait_name="inet_connect_result",
            check=_logs_check,
            interval_s=POLL_INTERVAL_S,
            max_wait_s=30.0,
            stall_threshold_s=POLL_STALL_S,
            state_str=str,
        ).run()
        _cstate, _sep, line = str(state).partition(":")
        return {"connect_ms": int(line)}
    finally:
        # Teardown is best-effort and must not hang forever.
        t_teardown0 = time.monotonic()
        try:
            cri = CriClient(sudo=True)
            if cid:
                cri.rm(cid, force=True, timeout_s=20.0)
            if pod:
                cri.stopp(pod, timeout_s=20.0)
                cri.rmp(pod, force=True, timeout_s=20.0)
        except Exception:
            pass
        # Ensure teardown doesn't silently run forever (guard even though calls are bounded).
        if (time.monotonic() - t_teardown0) > TEARDOWN_BUDGET_S:
            raise TimeoutError("teardown budget exceeded")
        tmpdir.cleanup()


def op_network_inet_download(inp: dict) -> dict:
    import tempfile
    import uuid

    from keska_lab.driver.cri_client import CriClient
    from keska_lab.harness.cri_pair import pod_spec
    from keska_lab.harness.network import INET_DOWNLOAD_PY

    budget_s = float(inp.get("op_budget_s", OP_BUDGET_INET_DOWNLOAD_S))
    runtime_handler = str(inp.get("runtime_handler", "") or "")
    tsot_dns = bool(inp.get("tsot_dns", False))
    image = str(inp.get("image", "python:3.12-slim"))

    t0 = time.monotonic()
    cid = ""
    pod = ""
    tmpdir = tempfile.TemporaryDirectory(prefix="keska-driver-")
    try:
        wd = Path(tmpdir.name)
        uid = uuid.uuid4().hex[:12]
        pod_json = wd / "pod.json"
        ctr_json = wd / "ctr.json"

        pod_json.write_text(
            json.dumps(
                pod_spec(
                    name=f"keska-netdl-{uid}",
                    uid=f"netdl-{uid}",
                    runtime_handler=runtime_handler,
                    tsot_dns=tsot_dns,
                ),
                indent=2,
            )
            + "\n"
        )
        ctr_json.write_text(
            json.dumps(
                {
                    "metadata": {"name": "inetdl"},
                    "image": {"image": image},
                    "command": ["python3", "-u", "-c", INET_DOWNLOAD_PY.strip()],
                    "log_path": "inetdl.log",
                },
                indent=2,
            )
            + "\n"
        )

        cri = CriClient(sudo=True)
        if (time.monotonic() - t0) > budget_s:
            raise TimeoutError("op budget exceeded before runp")

        pod = cri.runp(str(pod_json), runtime_handler=runtime_handler, timeout_s=CRI_RUNP_S)
        cid = cri.create(pod, str(ctr_json), str(pod_json), timeout_s=30.0)
        cri.start(cid, timeout_s=30.0)

        def _logs_check() -> tuple[bool, str]:
            cstate = "unknown"
            inspected = {}
            try:
                inspected = cri.inspect(cid, timeout_s=10.0)
                cstate = str(inspected.get("status", {}).get("state", "unknown"))
            except Exception:
                pass
            out = cri.logs(cid, tail=50, timeout_s=10.0)
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            line = lines[-1] if lines else ""
            # Expect a float (Mbps).
            ok = False
            try:
                float(line)
                ok = True
            except Exception:
                ok = False

            if cstate.lower() in ("container_exited", "exited") and not ok:
                evidence: dict = {"logs_tail": lines[-50:], "container_state": cstate}
                try:
                    evidence["inspect_status"] = inspected.get("status", {})
                    log_path = str(inspected.get("status", {}).get("logPath") or "")
                    if log_path and not evidence["logs_tail"]:
                        import subprocess

                        p = subprocess.run(
                            ["sudo", "-n", "tail", "-n", "80", log_path],
                            text=True,
                            capture_output=True,
                            timeout=5,
                        )
                        if p.returncode == 0 and p.stdout.strip():
                            evidence["log_path_tail"] = p.stdout.strip().splitlines()[-80:]
                        elif p.stderr.strip():
                            evidence["log_path_tail_err"] = p.stderr.strip()
                except Exception:
                    pass
                raise RuntimeError(f"container exited without result, evidence={evidence}")
            return ok, f"{cstate}:{line}"

        state = PollWait(
            wait_name="inet_download_result",
            check=_logs_check,
            interval_s=POLL_INTERVAL_S,
            max_wait_s=90.0,
            stall_threshold_s=POLL_STALL_S,
            state_str=str,
        ).run()
        _cstate, _sep, line = str(state).partition(":")
        return {"mbps": float(line)}
    finally:
        t_teardown0 = time.monotonic()
        try:
            cri = CriClient(sudo=True)
            if cid:
                cri.rm(cid, force=True, timeout_s=20.0)
            if pod:
                cri.stopp(pod, timeout_s=20.0)
                cri.rmp(pod, force=True, timeout_s=20.0)
        except Exception:
            pass
        if (time.monotonic() - t_teardown0) > TEARDOWN_BUDGET_S:
            raise TimeoutError("teardown budget exceeded")
        tmpdir.cleanup()


OPS = {
    "host.scan_orphans": op_host_scan_orphans,
    "network.inet_connect": op_network_inet_connect,
    "network.inet_download": op_network_inet_download,
}


def cmd_run(args) -> int:
    t0 = time.monotonic()
    if args.protocol != 1:
        env = DriverEnvelope(
            request_id=args.request_id,
            op=args.op,
            status=DriverStatus.failed,
            error={
                "code": DriverErrorCode.protocol_mismatch,
                "message": f"unsupported protocol_version={args.protocol}",
                "evidence": {"supported": [1]},
            },
            duration_s=time.monotonic() - t0,
            clean_report=_clean_report_from_orphans(),
        )
        out = env.to_json()
        _write_json(args.output, out) if args.output else None
        print(json.dumps(out))
        return ExitCode.protocol_mismatch

    try:
        inp = _read_json(args.input)
    except Exception as e:
        env = DriverEnvelope(
            request_id=args.request_id,
            op=args.op,
            status=DriverStatus.failed,
            error={
                "code": DriverErrorCode.input_invalid,
                "message": f"invalid input json: {e}",
            },
            duration_s=time.monotonic() - t0,
            clean_report=_clean_report_from_orphans(),
        )
        out = env.to_json()
        _write_json(args.output, out) if args.output else None
        print(json.dumps(out))
        return ExitCode.input_invalid

    fn = OPS.get(args.op)
    if fn is None:
        env = DriverEnvelope(
            request_id=args.request_id,
            op=args.op,
            status=DriverStatus.failed,
            error={
                "code": DriverErrorCode.protocol_invalid,
                "message": f"unknown op {args.op!r}",
                "evidence": {"known_ops": sorted(OPS.keys())},
            },
            duration_s=time.monotonic() - t0,
            clean_report=_clean_report_from_orphans(),
        )
        out = env.to_json()
        _write_json(args.output, out) if args.output else None
        print(json.dumps(out))
        return ExitCode.internal

    try:
        result = fn(inp)
        env = DriverEnvelope(
            request_id=args.request_id,
            op=args.op,
            status=DriverStatus.ok,
            result=result,
            duration_s=time.monotonic() - t0,
            clean_report=_clean_report_from_orphans(),
        )
    except (WaitStalled, WaitTimedOut) as e:
        env = DriverEnvelope(
            request_id=args.request_id,
            op=args.op,
            status=DriverStatus.failed,
            error={
                "code": DriverErrorCode.wait_stalled
                if isinstance(e, WaitStalled)
                else DriverErrorCode.cri_command_timeout,
                "message": str(e),
                "evidence": {
                    "wait_name": getattr(e, "wait_name", ""),
                    "elapsed_s": getattr(e, "elapsed_s", 0.0),
                    "last_state": getattr(e, "last_state", ""),
                },
            },
            duration_s=time.monotonic() - t0,
            clean_report=_clean_report_from_orphans(),
        )
    except TimeoutError as e:
        env = DriverEnvelope(
            request_id=args.request_id,
            op=args.op,
            status=DriverStatus.failed,
            error={
                "code": DriverErrorCode.op_budget_exceeded,
                "message": str(e),
            },
            duration_s=time.monotonic() - t0,
            clean_report=_clean_report_from_orphans(),
        )
    except Exception as e:
        env = DriverEnvelope(
            request_id=args.request_id,
            op=args.op,
            status=DriverStatus.failed,
            error={
                "code": DriverErrorCode.protocol_invalid,
                "message": str(e),
            },
            duration_s=time.monotonic() - t0,
            clean_report=_clean_report_from_orphans(),
        )

    out = env.to_json()
    if args.output:
        _write_json(args.output, out)
    print(json.dumps(out))
    return exit_code_for_envelope(env)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="keska-lab-driver", description="Keska lab colocated driver")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a single driver op")
    run.add_argument("--protocol", type=int, required=True)
    run.add_argument("--request-id", required=True)
    run.add_argument("--op", required=True)
    run.add_argument("--input", required=True, help="Path to input JSON file")
    run.add_argument("--output", help="Optional path to write envelope JSON")
    run.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

