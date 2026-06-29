# 014 — UringFixedFile without register_files

**Feature deleted 2026-06-29** (E1). This note documents the pre-deletion bug investigation.

## Symptom

With `UringFixedFile: true`, sandboxes hung (300s+ TTI), host RSS on `quark boot` reached ~1–3 GB vs ~21 MB baseline, io_uring completions failed.

## Cause

`host_uring.rs` set `FIXED_FILE` on all SQEs but never called `register_files` / `register_files_sparse`. The SQE fd field still held raw host fds while the kernel expected fixed-table indices — invalid submissions and runaway failure under load.

`UringMgr` pre-allocated a 16K `i32` table but never registered it with io_uring; `Close()` incorrectly `close()`d entries it did not own.

After the register_files fix, `HostSubmit()` still deadlocked: it held `URING.lock()` while building SQEs, and `RegisterHostFd()` tried to take `URING.lock()` again for `register_files_update`. First I/O on each fd hung the KIO thread; guest vCPUs spun at 100% CPU.

## Fix

- `UringMgr`: sparse table via `register_files_sparse`, lazy `register_files_update` per host fd, `UnregisterHostFd` on close, no spurious `close()` on table teardown.
- `host_uring.rs`: map host fds to `types::Fixed(index)` + `FIXED_FILE` only after registration; special fds (`< 0`, e.g. `AT_FDCWD`) stay `types::Fd`.
- `host_uring.rs`: pre-register host fds in `HostSubmit()` **before** taking `URING.lock()`; SQE build uses `LookupHostFd` only.
- `FdInfoIntern::Close`: unregister from fixed table before closing OS fd.

## Files

- `qvisor/src/vmspace/uringMgr.rs`
- `qvisor/src/vmspace/host_uring.rs`
- `qvisor/src/vmspace/HostFileMap/fdinfo.rs`

## Verify

```bash
make -C qvisor release CARGO_FEATURES=experimental-uring-fixed-file
# lab: deploy UringFixedFile=true, keska-lab-group5 UringFixedFile --experimental-only
# expect: light TTI ~50ms, memory_idle ~21MB (+ small overhead), full IO completes
```
