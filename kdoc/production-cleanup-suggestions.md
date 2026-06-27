# Production cleanup suggestions

Suggestions for trimming, hardening, and simplifying the Quark tree before a production push. This is a **review checklist**, not a committed plan — prioritize against your actual deployment shape (bare OCI CLI vs containerd shim vs K8s + TSOT).

---

## What “production” means here

**Ship path today** (`make install`):

| Artifact | Source |
|----------|--------|
| `quark`, `containerd-shim-quark-v1` | `qvisor/` |
| `qkernel.bin` | `qkernel/` |
| `vdso.so` | `vdso/` |
| `/etc/quark/config.json` | root `config.json` |

Everything else is optional, experimental, or dev-only unless you explicitly productize it.

---

## Repository map: core vs experiment vs dev

### Core (keep, harden)

- **`qkernel/`**, **`qvisor/`**, **`qlib/`** (shared via symlinks from both crates)
- **`vdso/`**, **`config.json`**, root **`makefile`**
- **`sdk/python/`** — small gRPC client for `quarkd` (if daemon API is in scope)

### Platform / optional features (gate or split repo)

| Tree | Purpose | Default | Suggestion |
|------|---------|---------|------------|
| **`qservice/`** | Node agent, TSOT CNI, scheduler, gateway | Off (`EnableTsot: false`) | Separate “Quark Platform” build; not needed for `quark create` OCI |
| **`qserverless/`** | FaaS / serverless stack | Not in `make install` | **Defer** — multiple `unimplemented!()` stubs (`nm_store.rs`, `package.rs`, etc.) |
| **`rdma_cli/`**, **`rdma_srv/`** | Legacy RDMA socket path | Off (`EnableRDMA: false`) | **Defer or archive** unless a customer needs it; TSOT is the documented K8s networking path |
| **`cudaproxy/`** | CUDA LD_PRELOAD shim | `make cuda_all` only | Optional GPU SKU; keep behind `cuda` feature |
| **SEV-SNP** | Confidential compute | `CCMode: None` | Keep behind `snp` feature; aarch64/SNP still have open `todo!()` paths |

### Dev / benchmark only (do not ship)

- **`lab/`** — Keska harness; deploys bench config to remote hosts
- **`test/`** — manual C/Rust/Java/CUDA probes; not wired into root `makefile`
- **`benchmark/`** — standalone bench scripts
- **`minikube_install.sh`** — one-off bootstrap

### Docs / assets to thin

- **`doc/`** — mix of markdown + large binaries (PDF, PPTX, XLSX); archive binaries externally
- **`kdoc/bugs/`** — useful internally; fix duplicate numbering (`002-*`, `003-*` each appear twice)
- **`qservice/cmd.txt`**, **`env.txt`** — long runbooks; move into `kdoc/` as structured guides
- **`scripts/`** — empty; delete or restore intentionally

---

## Remove or defer (code reduction)

### Already-identified dead weight (post–file-I/O fastpath)

These survived the FileBuf removal and can go next:

| Item | Location | Why |
|------|----------|-----|
| `ENABLE_BUFF_IO` | `qlib/config.rs` | Constant `false`, zero readers |
| `AllocIOBuf` / `IsIOBuf` / IO heap | `qlib/mem/list_allocator.rs`, `qvisor/src/heap_alloc.rs`, `qkernel/src/kernel_def.rs` | No callers; **~1 GB guest VA** still reserved (`MemoryDef::IO_HEAP_SIZE` in `qlib/linux_def.rs`) |
| `IOBufWrite` qcall | `qvisor/src/vmspace/mod.rs`, `HostFileMap/fdinfo.rs` | Defined, never dispatched |
| Orphan `data_buff.rs` | `qlib/kernel/data_buff.rs`, `qkernel/src/data_buff.rs` | Not in module tree; `BufMgr` commented out in `qlib/range.rs` |

**Win:** less guest memory footprint, fewer allocator code paths, clearer io_uring-only story.

### Half-maintained config paths

| Flag | Shipped `config.json` | Rust `Default` | Action |
|------|----------------------|----------------|--------|
| `MmapRead` | `false` | `true` | **Remove flag + branch** in `hostinodeop.rs` *or* pick one behavior and delete the other |
| `UringFixedFile` | `false` | `false` | ~29 branch copies in `qvisor/src/vmspace/host_uring.rs` — remove or `cfg(feature)` |
| `UringStatx` | `false` | `false` | Single use in `qlib/kernel/fs/host/util.rs` — remove or document |
| `EnableRDMA` | `false` | `false` | Large socket/RDMA surface in `qlib/kernel/socket/hostinet/socket.rs` — feature-gate entire subsystem |
| `KernelPagetable` | `false` | `false` | Mostly commented code in `qkernel/src/interrupt/x86_64/mod.rs` — implement or delete |
| `HiberODirect` | `true` | `true` | Hibernate-only; keep only if hibernate is a product feature |

### Commented / orphan modules

- `qlib/mod.rs` — commented `macros`, `Process`, `uring`
- `qlib/kernel/memmgr/mod.rs` — commented `buf_allocator`
- `#![allow(dead_code)]` on **`qvisor/src/main.rs`** and **`qkernel/src/lib.rs`** — masks real dead code; remove attribute and fix warnings

### Naming confusion (not duplicate code)

- **`qlib/kernel/Kernel.rs`** — `HostSpace`, hypercalls (active)
- **`qlib/kernel/kernel/kernel.rs`** — guest `Kernel` struct (active)

Rename `Kernel.rs` → `hostspace.rs` (or similar) to stop onboarding mistakes.

### aarch64

`qlib/kernel/arch/aarch64/` has incomplete signal/FPU/pagetable work. **Defer** from production SKU until CI covers it; README already treats it as preliminary.

---

## Security

### Defaults that should flip for production

| Setting | Current shipped default | Risk | Suggestion |
|---------|-------------------------|------|------------|
| **`DisableCgroup: true`** | `config.json`, `Config::default()`, lab template | No CPU/mem/pid limits; skips cgroup install in `qvisor/src/runc/container/container.rs` | **Default `false`**; opt-out only for local dev |
| **`Sandboxed: false`** | Same | Multi-container sandbox semantics disabled; shim behavior differs | Document per deployment; consider `true` for containerd shim |
| **`CopyDataWithPf: true`** | Shipped config; Rust default `false` | Page-fault-driven copy from guest mappings — review for hostile workloads | Benchmark before enabling in hardened profiles |
| **`DebugLevel` / logging** | `Error` in config | OK | Ensure release builds cannot enable trace via env alone |

### Crash instead of fail-closed

- **Config load** — `qvisor/src/runc/cmd/cmd.rs` uses `.expect("configuration wrong format")` on bad JSON. Validate, merge with safe defaults, or refuse start with a clear error.
- **CC + TSOT** — `todo!()` in `qlib/kernel/quring/uring_mgr.rs` when confidential compute is active. Disable incompatible combos at config load time (partial check exists in `cmd.rs` for TSOT+CCMode string; extend to runtime CC).
- **SEV-SNP without `snp` build** — panics in `qvisor/src/runc/runtime/vm.rs`; return error at startup.

### Release build hygiene

- **`qkernel/Cargo.toml`** — `release_max_level_trace` allows trace logs in release guest kernel. Drop for production builds.
- **`PerfDebug: true`** in Rust `Config::default()` but `false` in shipped JSON — silent behavior change if config file is missing.

### Shim / runtime identity

- `qvisor/src/main.rs` — `ShimMode` selects containerd shim vs CLI; verify `"io.containerd.empty.v1"` runtime string is intentional for your registry/shim wiring.

---

## Performance

### Config alignment (avoid “missing config.json” surprises)

Shipped [`config.json`](../config.json) differs from [`qlib/config.rs`](../qlib/config.rs) `Default` on several hot-path flags:

| Flag | Shipped | Rust default | Notes |
|------|---------|--------------|-------|
| `EnableAIO` | `true` | `false` | `qkernel/src/syscalls/sys_aio.rs` |
| `EnableInotify` | `true` | `false` | `qlib/kernel/fs/dirent.rs` |
| `CopyDataWithPf` | `true` | `false` | `task_usermem.rs` |
| `TlbShootdownWait` | `true` | `false` | TLB shootdown path |
| `MmapRead` | `false` | `true` | File read path |
| `PerfDebug` | `false` | `true` | Print timing overhead |

**Suggestion:** one “production profile” struct or documented table; make Rust defaults match shipped JSON so behavior is predictable when config is absent.

### Memory

- Reclaim **IO heap** after dead-code removal (see above).
- Review **`KernelMemSize: 24`** GB default — right-size per instance class.

### io_uring path (current baseline)

- **`UringIO: true`** — keep; this is the lab-validated path after FileBuf fastpath removal.
- Trim **`UringFixedFile`** dead branches if not on a roadmap.

### Network (only if TSOT/RDMA enabled)

- **`qlib/kernel/socket/hostinet/socket.rs`** — many TODOs (IPv6, hardcoded addresses, port exhaustion). Not hot for bare OCI; matters for Node.js + TSOT workloads.

---

## Simplify maintainability

### Build targets

Split root `makefile` intent:

```text
make runtime    # qkernel + qvisor + vdso  → production
make platform   # qservice (+ qserverless if ever productized)
make gpu        # cudaproxy + cuda qvisor feature
make bench      # lab/ (out of tree install)
```

Today `cleanall` already touches qservice, qserverless, rdma_* — document that **`make install` ≠ full tree**.

### Toolchain

- Root `makefile` pins **`nightly-2024-07-01`**; README may still mention an older nightly. Unify and pin in one place (rust-toolchain.toml?).

### Config schema

Add **`kdoc/config-flags.md`** (or extend this doc) with one table:

| Flag | Production | Lab/dev | Owner file | Remove? |
|------|------------|---------|------------|---------|

All ~30 fields in `qlib/config.rs` — avoids rediscovering flags via grep.

### Documentation consolidation

| Keep in repo | Move / archive |
|--------------|----------------|
| `README.md`, `kdoc/quarkd-daemon.md`, `kdoc/sdk/python.md`, `lab/README.md` | `doc/*.pdf`, `doc/*.pptx`, stale `doc/roadmap.md` |
| `kdoc/bugs/` (renumber duplicates) | Duplicate SDK readme if redundant |
| `doc/k8s_setup.md`, `doc/perf_test.md` | `qservice/cmd.txt` → structured kdoc |

---

## Suggested phases

### Phase 1 — Quick wins (low risk)

1. Delete IO-buffer dead code (`ENABLE_BUFF_IO`, `AllocIOBuf`, `IOBufWrite`, orphan `data_buff.rs`, IO heap reservation).
2. Align Rust `Config::default()` with shipped `config.json`.
3. Remove `release_max_level_trace` from release guest kernel.
4. Fix `kdoc/bugs/` duplicate numbers; archive binary assets under `doc/`.
5. Document core vs experimental trees in README (short table).

### Phase 2 — Production hardening

1. Default **`DisableCgroup: false`**; validate cgroups in CI.
2. Harden config load (no panic on malformed JSON).
3. Disable or error on incompatible flag combos (CC + TSOT, SNP without feature build).
4. Remove `#![allow(dead_code)]` from qvisor/qkernel; burn down warnings.

### Phase 3 — Scope cuts (product decision)

1. Feature-gate or move **`qserverless/`**, **`rdma_*`** out of main repo.
2. Remove **`MmapRead`** or **`UringFixedFile`** if not on roadmap.
3. **`qservice/`** as separate release artifact for K8s-only customers.
4. Complete or drop **aarch64** and **hibernate** paths.

### Phase 4 — Performance pass (after scope is stable)

1. Re-benchmark with lab `full` + `db` suites on hardened defaults.
2. Socket/TSOT TODO cleanup if K8s networking is in scope.
3. Profile `CopyDataWithPf` vs alternatives for your workload mix.

---

## Lab harness note

`lab/` is **not** shipped but currently mirrors production defaults (`DisableCgroup: true`, etc.). When production defaults change, update [`lab/src/keska_lab/setup/quark_config.py`](../lab/src/keska_lab/setup/quark_config.py) so benchmarks stay representative — or add an explicit `production` profile in the harness.

---

## Related internal docs

- Postgres OCI / PGDATA: [bugs/004-postgres-oci-pgbench.md](bugs/004-postgres-oci-pgbench.md)
- TTI / harness parity: [bugs/012-tti-counts-teardown.md](bugs/012-tti-counts-teardown.md)
- Lab measurement philosophy: [lab/README.md](../lab/README.md)

---

*Generated after FileBuf fastpath removal (Jun 2026). Revisit when deployment target (OCI-only vs shim vs full K8s platform) is fixed.*
