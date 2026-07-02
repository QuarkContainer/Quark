# Containers and runtime

A guided tour from industry-standard container plumbing to Quark’s implementation.

Read in order if you are new; jump to a chapter if you already know the basics.

| # | Chapter | You will learn |
|---|---------|----------------|
| 1 | [Layered stack](01-layered-stack.md) | Where runc, shim, containerd, and kubelet sit |
| 2 | [OCI runtime & image spec](02-oci-runtime-spec.md) | Bundles, `config.json`, lifecycle, CRI annotations |
| 3 | [What runc does](03-what-runc-does.md) | Foreground vs detached, why detached matters |
| 4 | [What a shim does](04-what-shim-does.md) | Why shims exist; stdio, exit codes, attach |
| 5 | [Quark binary entry](05-quark-binary-entry.md) | `quark` vs `containerd-shim-quark-v1`, `Sandboxed` config |
| 6 | [Quark runc internals](06-quark-runc-internals.md) | Our `runc/` tree, VM, ucall, modules |
| 7 | [CRI pod lifecycle](07-cri-pod-lifecycle.md) | `runp` → create → start → exec in Quark |
| 8 | [Shim Task API](08-shim-task-api.md) | Implemented RPCs, gaps, backlog |

**External reference (excellent):** Ivan Velichko’s [Implementing Container Runtime Shim: runc](https://iximiuz.com/en/posts/implementing-container-runtime-shim/) — complements chapters 3–4.

**Code roots:** `qvisor/src/runc/`, `qvisor/src/main.rs`, `qlib/config.rs`.

---

## Beyond this series (stubs)

| Topic | Stub |
|-------|------|
| Guest kernel (qkernel) | [`../guest-kernel/README.md`](../guest-kernel/README.md) |
| Host VMM (vmspace, KVM) | [`../vmm/README.md`](../vmm/README.md) |
| TSOT networking | [`../networking-tsot/README.md`](../networking-tsot/README.md) |
| Confidential compute | [`../confidential-compute/README.md`](../confidential-compute/README.md) |
