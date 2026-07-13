from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import grpc


class TsotError(RuntimeError):
    pass


def _load_proto_stubs(proto_root: Path):
    """Generate na_pb2 / na_pb2_grpc under a stable package path.

    We do this at runtime to avoid committing generated files, but still keep it deterministic:
    - codegen output path is in repo under lab/src/keska_lab/tsot/_gen
    - only runs when stubs are missing
    """
    gen_dir = Path(__file__).resolve().parent / "_gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    pb2 = gen_dir / "na_pb2.py"
    pb2_grpc = gen_dir / "na_pb2_grpc.py"
    if pb2.exists() and pb2_grpc.exists():
        # grpcio-tools generates `na_pb2_grpc.py` which imports `na_pb2` as a
        # top-level module. Keep it simple: add gen_dir to sys.path and import.
        import sys

        gen_dir_str = str(gen_dir)
        if gen_dir_str not in sys.path:
            sys.path.insert(0, gen_dir_str)
        import na_pb2  # type: ignore
        import na_pb2_grpc  # type: ignore

        return na_pb2, na_pb2_grpc

    # Generate stubs.
    from grpc_tools import protoc  # type: ignore

    proto_file = proto_root / "na.proto"
    if not proto_file.exists():
        raise TsotError(f"missing na.proto at {proto_file}")

    args = [
        "protoc",
        f"-I{proto_root}",
        f"--python_out={gen_dir}",
        f"--grpc_python_out={gen_dir}",
        str(proto_file),
    ]
    rc = protoc.main(args)
    if rc != 0:
        raise TsotError(f"grpc_tools.protoc failed rc={rc}")

    # Ensure package init for import.
    (gen_dir / "__init__.py").write_text("# generated\n")

    return _load_proto_stubs(proto_root)


@dataclass(frozen=True)
class TsotClient:
    target: str = "127.0.0.1:8888"
    proto_root: Path | None = None
    timeout_s: float = 15.0

    def create_pod(self, *, pod_def: dict, config_map: dict | None = None) -> None:
        # Ensure TSOT state service is present; without it CRI runp can fail with ECONNRESET.
        from keska_lab.tsot.ensure import ensure_ss_running

        ensure_ss_running()
        proto_root = self.proto_root
        if proto_root is None:
            # default: repo layout on lab host
            proto_root = Path.home() / "Quark" / "qservice" / "qshare" / "proto"
        na_pb2, na_pb2_grpc = _load_proto_stubs(proto_root)

        pod_body = json.dumps(pod_def)
        req = na_pb2.CreatePodReq(pod=pod_body, configMap=json.dumps(config_map or {}))
        with grpc.insecure_channel(self.target) as ch:
            stub = na_pb2_grpc.NodeAgentServiceStub(ch)
            resp = stub.CreatePod(req, timeout=self.timeout_s)
        if getattr(resp, "error", ""):
            raise TsotError(f"CreatePod error: {resp.error}")

