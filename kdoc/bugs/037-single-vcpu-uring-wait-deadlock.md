# 037 — Single-vCPU io_uring Accept deadlock

**Tags:** upstream, cri, single-vcpu

## Symptom

CRI pause sandbox (`crictl runp` / `RunPodSandbox`) hangs ~120 s then `DeadlineExceeded`.
Guest reaches `Shared-space ready`; host `StartRootContainer` UCall never gets a response.

## Cause

Bug 035 runs pause sandboxes with one vCPU. `BootstrapTask` calls `SyncAccept` →
`UCall` → `Wait()` → `WaitFn` → `HYPERCALL_VCPU_WAIT` → `VcpuWait`. Two problems:

1. **`ProcessOnce` disabled**: `VcpuWait` had `Self::ProcessOnce(sharespace)` commented out.
   `ProcessOnce` calls `HostSubmit()` which flushes `submitq` SQEs to the kernel io_uring.
   Without it the Accept SQE was never submitted; `VcpuWait` blocked forever.

2. **`KERNEL_IO_THREAD.Wait()` is an infinite loop**: calling `HostSpace::IOWait()` from
   task context (inside `Wait()`) on single vCPU causes `KERNEL_IO_THREAD.Wait()` to run
   indefinitely, leaving the only vCPU stuck in the hypercall handler.

3. **Blocking `AcceptControl` HCall + loop = ControlMsgHandler starvation**: using a
   blocking `libc::accept` HCall for the accept means `BootstrapTask` re-enters the next
   accept immediately, leaving the freshly-created `ControlMsgHandler` task in the scheduler
   with no vCPU free to run it.

4. **`epoll_wait(-1)` after eventfd drain**: `VcpuWait` epollfd watches eventfd and
   `FD_NOTIFIER` but not the io_uring fd. Guest `Submit()` writes eventfd; host drains it
   on first wake, then blocks forever on `epoll_wait(-1)` if Accept CQE arrives later.

## Fix

- `qvisor/src/kvm_vcpu.rs` — `VcpuWait` calls `ProcessOnce` when `vcpuCnt == 1` (no
  dedicated IO vCPU). Use 1 ms `epoll_wait` timeout for single-vCPU so `ProcessOnce` polls
  io_uring completions. Create eventfd with `EFD_NONBLOCK`; tolerate `EAGAIN` on read.
  After `ProcessOnce`, re-check `Process()` before `epoll_wait` to avoid up to 1 ms delay
  when completions make a task runnable immediately.
- `qlib/kernel/taskMgr.rs` — guard `HostSpace::IOWait()` with `vcpuCnt > 1`. Single-vCPU
  tasks just spin with `pause` until `Wait()` switches to `WaitFn`.
- `qkernel/src/lib.rs` — `BootstrapTask` calls `InitLoader()` for `Sandboxed` regardless
  of `vcpuCnt` (restores plan fix).
- `qlib/kernel/boot/controller.rs` — keep `SyncAccept` (not blocking HCall) so
  `BootstrapTask` yields while waiting, allowing `ControlMsgHandler` to interleave.
- `qvisor/src/runc/runtime/sandbox_process.rs` — redirect boot stderr to
  `/var/log/quark/<id>.stderr` for panic capture.
- `qservice/qlet/tsot/pod_broker.rs` — `ProcessPodRegisterReq` Ok path returns
  `ErrCode::None` (was copy-paste `PodUidDonotExisit`).

## Files

- `qvisor/src/kvm_vcpu.rs`
- `qlib/kernel/taskMgr.rs`
- `qkernel/src/lib.rs`
- `qlib/kernel/boot/controller.rs`

## Verify

```bash
# On lab:
crictl runp /tmp/pod.json
crictl run /tmp/container.json <pod-id>
crictl exec <container-id> echo hello
```
