# AAA+ runtime roadmap

Operational plan to harden Quark from a development runtime to production grade. Work is ordered **Remove → Simplify → Harden → Build → Resolve experimental I/O**.

See also: [production-cleanup-suggestions.md](production-cleanup-suggestions.md) (dead-code inventory), [qkernel-linux-mechanisms-suggestions.md](qkernel-linux-mechanisms-suggestions.md) (guest kernel I/O gaps).

---

## Locked product decisions

| Topic | Decision |
|-------|----------|
| `MmapRead`, `UringStatx` | Keep behind Cargo features; resolve in Group 5 via full lab benchmarking → implement properly or delete |
| `UringFixedFile` (E1) | **Deleted** (2026-06-29) — no measurable gain at n=30; see [group5/e1-uring-fixed-file-decision-packet.md](group5/e1-uring-fixed-file-decision-packet.md) |
| `MmapRead` (E2) | **Recommend delete** — −70% io_read, coherency C7 RSS fail; see [group5/e2-mmap-read-decision-packet.md](group5/e2-mmap-read-decision-packet.md) |
| `qserverless/` | **Out of scope** for this roadmap — handled separately by product owner |
| `rdma_cli` / `rdma_srv`, TSOT | Keep — strategic performance differentiators |

---

## Group 1 — Remove

### R1: IO buffer dead code

Delete `IO_HEAP_SIZE` reservation, `AllocIOBuf` / `IsIOBuf`, `IOBufWrite` QCall, `ENABLE_BUFF_IO`.

**Test:** `cargo build`; `lab.bench_all("light", setup=False)`.

### R2: Orphan modules

Delete unreachable `data_buff.rs`; remove stale commented `mod` lines.

**Test:** `cargo build`.

### R3: Feature-gate experimental I/O

Cargo features (default off):

| Feature | Config flag | Crate |
|---------|-------------|-------|
| `experimental-mmap-read` | `MmapRead` | qkernel |
| `experimental-uring-statx` | `UringStatx` | qkernel |

Build experimental qkernel:

```bash
make -C qkernel release CARGO_FEATURES=experimental-mmap-read,experimental-uring-statx
```

**Test:** default `cargo build` succeeds; feature builds succeed.

### R4: Repo / install hygiene

- Split `make install` (release only) vs `make install-debug`
- Move `qservice/cmd.txt` → `kdoc/ops/quark-platform-runbook.txt`
- **Do not** touch `qserverless/`
- **Do not** delete `scripts/` (contains dev tooling)

---

## Group 2 — Simplify

| Batch | Change | Test |
|-------|--------|------|
| S1 | `rust-toolchain.toml`, drop `release_max_level_trace`, install split | compile |
| S2 | Align `Config::default()` with `config.json` | unit + lab light |
| S3 | Safe `NotImplementSyscall` → `SysNoSys` (vmsplice, openat2, pidfd_*, open_tree, move_mount) | unit + Go/glibc container |
| S4 | Config load / state machine panic → error | unit |
| S5 | Rename `qlib/kernel/Kernel.rs` → `hostspace.rs` | compile |

---

## Group 3 — Harden

| Batch | Change | Test |
|-------|--------|------|
| H1 | `DisableCgroup: false` default | lab light/db + cgroup limit test |
| H2 | Implement shim `stats()` from sandbox cgroup | `crictl stats` / lab |
| H4 | `PR_SET_NO_NEW_PRIVS` on sandbox child when OCI requests it | compile + container start |

---

## Group 4 — Build (later)

Prometheus endpoint, gRPC mTLS, TSOT NetworkPolicy, OOM events, systemd units, structured logging — see priority table in plan batches B1–B6.

---

## Group 5 — Resolve experimental I/O (last)

Run after Groups 1–3. **Delete or promote only after your explicit approval** — present a full decision packet first.

Lab harness in [lab/src/keska_lab/setup/quark_config.py](../lab/src/keska_lab/setup/quark_config.py) and A/B runner `keska-lab-group5`:

```bash
keska-lab-group5 MmapRead
keska-lab-group5 UringStatx
keska-lab-coherency              # C1–C5,C7 (default); add --include-c6 for 30 min soak
```

```python
from keska_lab.setup.quark_config import bench_config_json, experimental_config_json

bench_config_json()
experimental_config_json("MmapRead")
experimental_config_json("UringStatx")
```

### Global veto gates (every flag)

Baseline vs experimental, **≥5 iterations** per suite (`light`, `full`, `db`). Any veto failure → does not promote (recommend delete in packet; no code removal until you approve).

| Metric | Suite | Threshold |
|--------|-------|-----------|
| `tti_ms` | light | p50 ≤ baseline +5% |
| `tti_under_load_ms` | light | p50 ≤ baseline +5% |
| `memory_idle_rss_mb` | light, db | ≤ baseline +3% and +8 MB |
| `memory_while_paused_rss_mb` | light | ≤ baseline +3% and +8 MB |
| `pause_ms` / `resume_ms` | light | p50 ≤ baseline +10% |

### E2: `MmapRead`

**Build:** `make qkernel_release CARGO_FEATURES=experimental-mmap-read`

**Implement bar:** ≥8% io_read or ≥5% pgbench_tps; coherency C1–C7; postgres ≥30 min.

### E3: `UringStatx`

**Build:** `make qkernel_release CARGO_FEATURES=experimental-uring-statx`

**Implement bar:** stat micro-bench p50/p99 ≤ baseline; no `light` regression.

---

## Priority table

| Priority | Batches |
|----------|---------|
| P0 | R1, R2, S1, S3, S4, S5 |
| P1 | R3, R4, S2, H1, H2, H4 |
| P2 | H3, H5, B1, B4 |
| P3 | H6, B2, B3, B5, B6 |
| P4 | E1, E2, E3 |

---

## Anti-patterns

- In-guest slot caches on sync UCall I/O (FileBuf lesson)
- Async writes before io_uring completion
- Enabling experimental flags in production `config.json` without passing Group 5 bar
