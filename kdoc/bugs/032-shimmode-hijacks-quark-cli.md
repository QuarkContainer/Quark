# 032 — ShimMode config hijacks quark CLI

**Tags:** upstream, config

## Symptom

After deploying CRI+TSOT config (`ShimMode=true`), `quark list` and direct OCI fail — process enters containerd shim instead of CLI.

## Cause

`main.rs` routed to `containerd_shim::run` whenever `ShimMode` was true in `/etc/quark/config.json`, even for `quark list`.

## Fix

Enter shim only when argv0 is `containerd-shim-*`. CRI pod behavior uses **`Sandboxed`** in config. The **`ShimMode` JSON field was removed** from `config.json` / `qlib/config.rs` (2026-07-02).

## Files

- `qvisor/src/main.rs`
- `qlib/config.rs`, `config.json` (ShimMode removed)

## Verify

```bash
sudo quark list   # CLI works regardless of config
keska-lab-cri-bisect --bug 032
```
