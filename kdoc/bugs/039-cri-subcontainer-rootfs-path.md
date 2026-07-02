# 039 — CRI subcontainer guest root path mismatch

**Tags:** upstream, cri

## Symptom

Pod sandbox VM boots; workload `crictl run` fails with guest panic:

```
InitRootfs fail: SysError(2)
```

in `loader::StartSubContainer`.

## Cause

Host `MountContainerFs` bind-mounts the OCI rootfs to
`/var/lib/quark/<sandbox>/<containerId>`. After pivot that is `/<containerId>`.

`StartSubContainer` passed `Root: "/{id}/rootfs"` to the guest, so host inode
lookup failed with ENOENT.

## Fix

- `qvisor/src/runc/sandbox/sandbox.rs` — use `Root: "/{id}"` for Sandboxed
  subcontainers (same as direct OCI).

## Files

- `qvisor/src/runc/sandbox/sandbox.rs`

## Verify

```bash
sudo crictl run --no-pull container.json pod.json
sudo crictl exec $(sudo crictl ps -q | head -1) echo hello
```
