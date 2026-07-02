# 031 — CRI shim ignores containerd bundle path (Sandboxed mode)

**Tags:** upstream

## Symptom

`crictl run` with Quark CRI + `Sandboxed=true` fails:

```
open /{sandbox-id}/options.json: No such file or directory
```

## Cause

`ContainerFactory::Create` forced `bundle = "/{id}"` whenever `Sandboxed` was set. containerd 2.x podsandboxer passes the real bundle under `/run/containerd/...`.

## Fix

Only use the legacy `/{id}` bundle when `req.bundle` is empty.

## Files

- `qvisor/src/runc/shim/container.rs`

## Verify

```bash
keska-lab-network-preflight
```
