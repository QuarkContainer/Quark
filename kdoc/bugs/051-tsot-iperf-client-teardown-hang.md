# 051 — TSOT pod-to-pod iperf client hangs on teardown

## Symptom

Pod-to-pod iperf through TSOT relay reaches full throughput (~700+ Mbits/s on lab), but a **foreground** `crictl exec … iperf3 -c` often never exits. Server-side logs show a completed run; the client process blocks until killed. Lab network bench hit 120–180s SSH timeouts when the harness waited on client exit.

Secondary symptom: repeated readiness probes left iperf3 servers in **“busy running a test”**, causing false-negative readiness loops.

## Cause

1. **Relay fd leak (`conn_svc.rs`)** — After `HandlePeerConnect`, `TcpSvcConnection::Process` intentionally leaked the accepted relay `TcpStream` fd (`into_raw_fd`) to keep the TCP session open. SCM_RIGHTS already duplicates the fd to the server guest, so the extra relay copy prevented the server-side TCP half from closing. Clients waiting for FIN/results (iperf3) hung indefinitely.

2. **Async `PeerConnectNotify` (`pod_broker.rs`)** — `HandleNewPeerConnection` queued the notify via `EnqMsg`, racing with relay teardown and making fd lifetime harder to reason about.

3. **Harness server mode** — Readiness polling against a multi-client iperf3 server left it “busy” between attempts. Fixed with `iperf3 -s -1` (single client).

## Fix

**Platform**

- `HandleNewPeerConnection`: send `PeerConnectNotify` synchronously via `SendMsg`.
- `TcpSvcConnection::Process`: after `WriteConnResp(Ok)`, **drop** the relay stream instead of leaking the fd.

**Lab harness**

- Server: `iperf3 -s -1` (single client).
- Client: background exec (crictl blocks on exec teardown even after relay fd fix); poll server logs for throughput.
- Orphan-process scan/warn in `quark_cleanup` + suite runner (hygiene only).

## Files

- `qservice/qlet/tsot/conn_svc.rs`
- `qservice/qlet/tsot/pod_broker.rs`
- `lab/src/keska_lab/harness/network.py`
- `lab/src/keska_lab/setup/quark_cleanup.py`
- `lab/src/keska_lab/harness/runner.py`

## Verify

```bash
# Rebuild/sync na on lab, then:
cd lab && . .venv/bin/activate
KESKA_LAB_SKIP_REGISTRY_AUTH=1 python3 -c "
from keska_lab.session import LabSession
from keska_lab.profile import NodeProfile
lab = LabSession(profile=NodeProfile.quark_tsot())
lab.bench('network', n=1, setup=False)
"
# Expect sandbox_iperf_mbps ~700+ and total runtime < 60s

# Foreground iperf should exit within ~10s after platform fix:
# crictl exec client iperf3 -c $SERVER_IP -t 3 -f m
```
