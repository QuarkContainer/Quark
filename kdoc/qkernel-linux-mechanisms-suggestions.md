# QKernel vs Linux: missing mechanisms for production performance

Suggestions for guest-kernel mechanisms that Linux provides and QKernel largely lacks or stubs. Grounded in **Postgres lab work** (pgbench TPS gap vs Kata, bind-mount PGDATA, io_uring UCall read/write baseline) and a read of current syscall/VFS paths.

**Impact scale**

| Label | Meaning |
|-------|---------|
| **Critical** | Can change results by multiples (2×–10×+) for workloads that depend on it |
| **High** | Often 30–100%+ on targeted benchmarks; “feels broken” without it |
| **Medium** | Meaningful for specific apps or configs; incremental elsewhere |
| **Low** | Niche, polish, or already “good enough” |

**Complexity**

| Label | Rough effort |
|-------|----------------|
| **Low** | Days; mostly wire existing host/qvisor code |
| **Medium** | Weeks; new guest logic + tests |
| **High** | Months; coherency, security review, broad test matrix |
| **Very high** | Multi-quarter; new subsystem |

---

## Architectural gap (why Postgres hurt)

Linux gives applications a **unified page cache**: `read()` often hits RAM; `readahead` / `posix_fadvise` train the cache; `mmap` maps cache pages into the address space.

QKernel’s default host file path (`UringIO` + `hostinodeop.rs`) is roughly:

```text
guest read() → DataBuff in guest → io_uring UCall READ → host → CopyDataOutToIovs → app buffer
```

Every read pays a **VM exit / QCall** and at least **one extra copy**. There is no guest page cache and **hints to the host cache are dropped or no-ops**. That matches what we saw: postgres/pgbench and large sequential I/O stay far below native Linux/Kata even when storage is similar.

The removed FileBuf slot cache tried to patch this in-guest and failed (overhead > benefit). The lesson: adopt **Linux-shaped mechanisms** (cache, hints, mmap, zero-copy), not another ad-hoc guest buffer layer.

---

## Summary table

| # | Mechanism | Impact | Complexity | Primary apps | Main drawbacks |
|---|-----------|--------|------------|--------------|----------------|
| 1 | [Forward `fadvise` / `readahead` to host](#1-forward-fadvise--readahead-to-host) | **Critical** | **Low** | Postgres, MySQL, analytics, log tailers | Host cache ≠ guest; reads still copy unless mmap path used |
| 2 | [Mmap / shared mapping fast read path](#2-mmap--shared-mapping-fast-read-path) | **Critical** | **High** | Postgres (shared buffers mmap), databases, ML data loaders | Write coherency, invalidation, tmpfs/uring EINVAL edges |
| 3 | [Zero-copy / direct-to-user io_uring reads & writes](#3-zero-copy--direct-to-user-io_uring) | **High** | **Medium–High** | Postgres checkpoints, bulk load, nginx, object stores | Pinning guest pages, partial I/O, security on shared mem |
| 4 | [Guest page cache (fd + offset)](#4-guest-page-cache) | **High** | **High** | Any read-heavy DB or file server | Memory accounting, invalidation on write/truncate |
| 5 | [Honor `O_DIRECT` end-to-end](#5-honor-o_direct-end-to-end) | **High** | **Medium** | Postgres (`direct_io`), Kafka, Cassandra | No cache; alignment rules; harder on bind mounts |
| 6 | [Real `fdatasync` vs `fsync`](#6-real-fdatasync-vs-fsync) | **High** | **Low–Medium** | Postgres WAL, SQLite, etcd | Must preserve ordering; metadata sync semantics |
| 7 | [Guest `io_uring` (setup/submit)](#7-guest-io_uring) | **Critical** (if enabled in app) | **Very high** | Postgres 16+ AIO, Node 20+, custom servers | Huge attack surface; ring shared memory; ABI churn |
| 8 | [Functional `madvise` for file mappings](#8-functional-madvise-for-file-mappings) | **High** | **Low–Medium** | Postgres, JVM, jemalloc/tcmalloc | Over-prefetch; wrong hint → memory pressure |
| 9 | [Host `sendfile` / `copy_file_range` fast path](#9-host-sendfile--copy_file_range) | **Medium–High** | **Medium** | nginx, Caddy, backup tools, `cp` | Only helps socket/file combinations you fast-path |
| 10 | [`fallocate` modes beyond default](#10-fallocate-modes-beyond-default) | **Medium** | **Low** | Postgres tablespaces, sparse files, VM images | Punch-hole coherency with mmap |
| 11 | [Vectored I/O batching at QCall boundary](#11-vectored-io-batching-at-qcall-boundary) | **Medium** | **Medium** | All write-heavy apps; PG WAL batching | Latency vs throughput tradeoff |
| 12 | [Sequential read auto-readahead (kernel heuristic)](#12-sequential-read-auto-readahead) | **High** | **Medium** | Scans, log replay, `cat`, bulk ETL | Wrong detection on mixed workloads |
| 13 | [`sync_file_range` + write lifecycle tuning](#13-sync_file_range--write-lifecycle-tuning) | **Medium** | **Low** | Postgres bgwriter/checkpoint tuning | Easy to get ordering wrong |
| 14 | [Write combining / async write completion](#14-write-combining--async-write-completion) | **Medium** | **Medium** | WAL-heavy DBs, journaling FS users | Durability visibility; error propagation |
| 15 | [Epoll/accept already async — verify under load](#15-network-path) | **Low–Medium** | **Low** | Web servers, PG connection storms | TSOT/RDMA paths add complexity |

---

## Detailed suggestions

### 1. Forward `fadvise` / `readahead` to host

**Today**

- `SysFadvise64` (`qkernel/src/syscalls/sys_file.rs`) returns `Ok(0)` without doing anything.
- `SysReadahead` (`qkernel/src/syscalls/sys_read.rs`) always returns **`EINVAL`**.
- Host support **already exists**: `HostSpace::Fadvise` → qvisor `posix_fadvise` (`qlib/kernel/Kernel.rs`, `qvisor/src/vmspace/mod.rs`). No guest `readahead` QCall yet.

**Proposal**

- Wire `SysFadvise64` for regular host files to `HostSpace::Fadvise` on the backing `HostFd`.
- Add QCall + `SysReadahead` → host `readahead(2)`.

**Apps:** Postgres (`posix_fadvise` on sequential scans, bulk read), MySQL InnoDB, ClickHouse-style engines, any ETL.

**Drawbacks:** Populates **host** page cache only; guest `read()` still copies unless paired with mmap/zero-copy (#2–3). Postgres may still win a lot on sequential I/O because host read latency drops.

**Impact:** **Critical** for scan-heavy DB workloads (often largest gap vs Linux).  
**Complexity:** **Low** (days).

---

### 2. Mmap / shared mapping fast read path

**Today**

- `MmapRead` config path in `hostinodeop.rs` maps **2 MiB chunks** via `HostSpace::MMapFile` and copies from mapped host memory — disabled in shipped `config.json` (`MmapRead: false`).
- Generic `mmap(MAP_SHARED)` of host files works for mapping but **`read()` does not use the mapping** unless `MmapRead` is on.
- Postgres can use buffer-manager reads via `read()` even when data files are mmap-capable.

**Proposal**

- Treat **MAP_SHARED file mappings** as the primary read source when a vma covers the range (Linux behavior).
- Or re-enable and harden `MmapRead` (coherency with writes, truncate, msync).
- Prefer **host page cache + mmap** over per-read UCall.

**Apps:** Postgres (data file access patterns), SQLite, LMDB, memory-mapped analytics.

**Drawbacks:** **High** complexity — shared mapping coherency, write invalidation, `msync`/`fdatasync` interaction, CC/TEE writeback paths (`WritebackAllPages`).

**Impact:** **Critical** for DBs that mix mmap and read (or if shared_buffers uses mmap of data files).  
**Complexity:** **High** (months for production-grade coherency).

---

### 3. Zero-copy / direct-to-user io_uring

**Today**

- `IOURING.Read` reads into a guest **`DataBuff`**, then `CopyDataOutToIovs` (`hostinodeop.rs`).
- Same pattern on write (copy in, then UCall write).

**Proposal**

- Register guest user iovec pages with qvisor (pinned shared regions) and issue io_uring READ/WRITE **directly into app buffers** when safe.
- Fallback to current path for tmpfs (`EINVAL`), misaligned buffers, or CC mode.

**Apps:** Postgres bulk read/write, pg_dump/restore, nginx static files, S3-style gateways.

**Drawbacks:** Pinning/lifetime, short reads, security of guest-chosen addresses across the boundary.

**Impact:** **High** — removes one full copy on every I/O; often 1.5–2× on bandwidth-bound work.  
**Complexity:** **Medium–High**.

---

### 4. Guest page cache

**Today:** No cache keyed by `(host_fd, offset)`; removed FileBuf slots were a failed version of this.

**Proposal:** Small LRU of host-backed pages or cache host mmap chunks in guest shared memory with explicit invalidation on write/truncate/`fallocate`.

**Apps:** Read-heavy OLAP, repeated index scans, static asset servers.

**Drawbacks:** Memory limits in microVM, invalidation bugs, duplicates host cache unless carefully scoped.

**Impact:** **High** if host cache isn’t visible to guest reads; **Medium** if #1+#2 done well.  
**Complexity:** **High**.

---

### 5. Honor `O_DIRECT` end-to-end

**Today:** `O_DIRECT` is parsed in `qlib/kernel/fs/flags.rs` but **not consulted** in `hostinodeop` read/write paths (no grep hits on `.Direct` in FS code).

**Proposal:** Propagate to host `open` and use aligned buffer rules; bypass fadvise/cache paths; use host O_DIRECT semantics.

**Apps:** Postgres with `direct_io` / some tablespaces, Kafka, RocksDB, video pipelines.

**Drawbacks:** Strict alignment; bind-mount/tmpfs may not support; harder debugging.

**Impact:** **High** for direct-I/O configs; **Low** for default Postgres.  
**Complexity:** **Medium**.

---

### 6. Real `fdatasync` vs `fsync`

**Today**

- `SysDatasync` comment: *“just calls Fsync, which is a big hammer, but correct”* (`qkernel/src/syscalls/sys_sync.rs`).
- Under `UringIO`, `IOURING.Fsync(..., datasync=true)` exists — distinction may already reach host for that path; syscall layer still treats both as full file sync at the VFS layer.

**Proposal:** Ensure WAL path uses **`fdatasync`** semantics (skip metadata flush where Linux would), and that async write completion is visible before sync returns.

**Apps:** **Postgres** (every commit), SQLite, etcd, RocksDB WAL.

**Drawbacks:** Subtle durability differences if metadata assumed synced.

**Impact:** **High** for commit latency under sync-heavy load.  
**Complexity:** **Low–Medium**.

---

### 7. Guest `io_uring`

**Today:** `io_uring_setup` / submit → **`SysNoSys`** (`qkernel/src/syscalls/syscalls.rs`). Host qvisor uses io_uring internally; guests cannot.

**Proposal:** Optional passthrough or limited emulation (read/write/fsync on host fds only).

**Apps:** Postgres 16+ optional AIO subsystem, Node.js io_uring, high-QPS custom servers.

**Drawbacks:** **Very high** security and maintenance cost; SQ/CQ shared rings; opcode compatibility; seccomp bypass concerns.

**Impact:** **Critical** only if you enable io_uring in upstream apps; **Low** if all apps stick to sync syscalls.  
**Complexity:** **Very high**.

---

### 8. Functional `madvise` for file mappings

**Today**

- `SysMadvise`: `MADV_SEQUENTIAL`, `RANDOM`, `WILLNEED`, `NORMAL` are **commented no-ops** (`qkernel/src/syscalls/sys_mmap.rs`).
- `HostInodeOpIntern::MAdvise` **does** forward to host for **already mapped** file physical ranges.

**Proposal:** Implement vma-level hints; forward to host `madvise` on underlying file mappings; tie into readahead policy (#1, #12).

**Apps:** Postgres, JVM heap/off-heap, glibc malloc arenas, any mmap-heavy runtime.

**Drawbacks:** `MADV_WILLNEED` can spike memory; wrong hints hurt.

**Impact:** **High** for mmap-heavy DB and JVM workloads.  
**Complexity:** **Low–Medium**.

---

### 9. Host `sendfile` / `copy_file_range`

**Today**

- `SysSendfile` → guest `DoSplice` (`sys_splice.rs`) — pipe-style splice, not necessarily host `sendfile(2)`.
- `copy_file_range` → **`SysNoSys`**.

**Proposal:** For `regular_file → socket`, use host `sendfile`; for file→file, host `copy_file_range` (qvisor already has related libc wrappers for other ops).

**Apps:** nginx/Caddy static, CDN edges, backup/copy tools.

**Drawbacks:** Edge cases (non-blocking, partial counts, inotify); less relevant to Postgres.

**Impact:** **Medium–High** for web/static; **Low** for Postgres.  
**Complexity:** **Medium**.

---

### 10. `fallocate` modes beyond default

**Today:** `SysFallocate` rejects `mode != 0` with **`ENOTSUP`** (`sys_file.rs`); mode 0 goes through `inode.Allocate`.

**Proposal:** Forward `FALLOC_FL_KEEP_SIZE`, `PUNCH_HOLE`, `ZERO_RANGE` to host `fallocate` (qvisor has `fallocate` in `vmspace/mod.rs`).

**Apps:** Postgres tablespace growth, thin provisioning, VM/sparse images.

**Drawbacks:** Punch hole vs mmap coherency.

**Impact:** **Medium** for long-running DB production (space management).  
**Complexity:** **Low**.

---

### 11. Vectored I/O batching at QCall boundary

**Today:** `readv`/`writev` may decompose into per-iovec or per-chunk UCalls with buffer allocation in guest.

**Proposal:** Single UCall for contiguous host operations; coalesce small WAL writes where safe.

**Apps:** Postgres WAL append, write-heavy microservices.

**Drawbacks:** Latency buffering; error partiality.

**Impact:** **Medium**.  
**Complexity:** **Medium**.

---

### 12. Sequential read auto-readahead (kernel heuristic)

**Today:** No heuristic; apps must use `readahead`/`fadvise` (both broken/stubbed).

**Proposal:** In `hostinodeop` ReadAt, detect sequential stride and prefetch via host `readahead` or async uring reads — even without app hints.

**Apps:** Table scans, `pg_dump`, log replay, `cat`/`dd` benchmarks.

**Drawbacks:** False sequential detection on mixed I/O.

**Impact:** **High** as a safety net when apps don’t hint.  
**Complexity:** **Medium**.

---

### 13. `sync_file_range` + write lifecycle tuning

**Today:** `SyncFileRange` wired to host (`hostinodeop.rs`, `HostSpace::SyncFileRange`) — **better than readahead**.

**Proposal:** Document for PG tuning; ensure interaction with mmap writeback and `fdatasync` ordering.

**Apps:** Postgres checkpoint/bgwriter tuning.

**Impact:** **Medium**.  
**Complexity:** **Low** (mostly validation + docs).

---

### 14. Write combining / async write completion

**Today:** Synchronous UCall `Write` completion before return (post–FileBuf removal). Historical FileBuf async write had durability bugs.

**Proposal:** Async uring writes with **strict** completion before `write()` returns to guest (Linux-like), or explicit `O_DSYNC`/`fdatasync` barriers — not fire-and-forget.

**Apps:** WAL-heavy databases.

**Drawbacks:** Prior async write bugs (see removed fastpath); must not repeat.

**Impact:** **Medium** (throughput under parallel writers).  
**Complexity:** **Medium**.

---

### 15. Network path

**Today:** `AsyncAccept`, io_uring socket ops, epoll — relatively mature. Postgres is often **disk-bound** in lab, not accept-bound.

**Proposal:** Profile before investing; focus disk path first.

**Impact:** **Low–Medium** for PG; **High** for edge HTTP services.  
**Complexity:** **Low** for profiling; variable for fixes.

---

## Postgres-specific playbook (lab → production)

What likely dominated the lab gap:

1. **No host readahead/fadvise** → sequential scans and even pgbench hit cold host paths every time.
2. **Per-read QCall + copy** → amplifies bind-mount/tmpfs latency vs Kata’s block-backed stack.
3. **`fdatasync` / WAL** → commit path may sync more than Linux (`fdatasync` → full fsync semantics at VFS comment layer).
4. **Storage path** (documented in lab README) — Kata devmapper vs Quark bind-mount still differs; kernel fixes won’t equalize TPS alone.

**Recommended order for Postgres**

| Phase | Items | Expected effect |
|-------|--------|-----------------|
| **P0** | #1 Wire fadvise + readahead | Large win on scans / pgbench read phase |
| **P0** | #6 fdatasync semantics | Lower commit latency |
| **P1** | #3 Zero-copy uring OR #2 mmap read fast path | Bandwidth / CPU copy reduction |
| **P1** | #8 madvise for mapped regions | Helps if PG uses mmap buffers |
| **P2** | #5 O_DIRECT, #10 fallocate modes | Production configs / tablespaces |
| **P3** | #7 Guest io_uring | Only if PG AIO enabled in config |

Re-verify with `lab.bench_all("db")` and `full` after each phase; compare Quark vs Kata with **same PGDATA backing** when measuring kernel changes.

---

## What not to repeat

- **In-guest slot caches** on top of sync UCall I/O (removed FileBuf path) — added locks/copies without host cache integration.
- **Async write “fastpaths”** that return before uring completion — caused postgres crashes and silent data risks.
- **Config-only toggles** (`MmapRead`) without coherency tests — disabled in production config for good reason until hardened.

Prefer mechanisms Linux apps **already call** (`fadvise`, `readahead`, `mmap` coherency, `fdatasync`) over new Quark-specific APIs.

---

## Related docs

- [future-plans.md](future-plans.md) — storage/disk fairness, repo scope
- [bugs/004-postgres-oci-pgbench.md](bugs/004-postgres-oci-pgbench.md) — PGDATA harness
- [lab/README.md](../lab/README.md) — measurement parity and storage caveats

---

*Draft from QKernel source review + Postgres lab context (Jun 2026). Re-rank after profiling (`perf`, lab suites) on your target deployment.*
