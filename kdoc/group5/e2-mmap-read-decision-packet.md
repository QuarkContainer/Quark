# E2 — `MmapRead` decision packet (n=30 + coherency, 2026-06-29)

**Flag:** `MmapRead` (qkernel `experimental-mmap-read`)  
**Lab:** `lab@lab.keska.vpn`  
**Recommendation:** **Delete** — performance regression + C7 RSS fail; pending your explicit approval  
**A/B packet JSON:** `~/.keska-lab/results/group5/2026-06-29T09-56-26_MmapRead_decision_packet.json`  
**Coherency JSON:** `~/.keska-lab/results/group5/2026-06-29T12-24-40_MmapRead_coherency.json`

---

## Performance A/B (p50, n=30)

| Metric | Baseline | Experimental | Delta | Gate |
|--------|----------|--------------|-------|------|
| light `tti_ms` | 53 ms | 59 ms | +11.3% | **FAIL** |
| full **`io_read_mib_s`** | **6144** | **1843** | **−70%** | **FAIL** |
| db **`pgbench_tps`** | **328.9** | **319.2** | **−2.9%** | **FAIL** |
| full `io_write_mib_s` | 1010 | 929 | −8.1% | — |
| full `io_concurrent_read` | 1936 | 1401 | −27.6% | — |

Experimental io_read clustered at **1843 MiB/s** (identical at n=5 and n=30) vs baseline **4.6–6.2 GiB/s** — systematic mmap-read path ceiling, not noise.

---

## Coherency C1–C7 (2026-06-29, `keska-lab-coherency`)

| Case | Scenario | Result |
|------|----------|--------|
| C1 | Host write → guest read() | **PASS** |
| C2 | Guest write → guest read() | **PASS** |
| C3 | Truncate while mapped | **PASS** |
| C4 | Concurrent host write + guest read | **PASS** |
| C5 | mmap vs read() same region | **PASS** |
| C6 | pgbench ≥30 min soak | **Not run** (use `--include-c6`) |
| C7 | 1000× open/read/close RSS | **FAIL** (+19.8%: 29.75→35.65 MB) |

**Coherency verdict:** basic read/mmap/truncate coherency holds (C1–C5), but **chunk cache RSS growth** fails C7 — consistent with per-read `MMapChunk` without release on hot paths.

---

## Why delete (even though C1–C5 pass)

1. **Wrong tradeoff for lab workloads** — replaces io_uring with mmap+copy; ~3× io_read regression, global TTI hit.
2. **Implement bars failed** — need ≥8% io_read or ≥5% pgbench; got large negative deltas.
3. **C7 RSS fail** — mappable 2 MiB chunks accumulate under repeated open/read/close.
4. **Not the roadmap target** — Linux-like “read from existing MAP_SHARED vma” is unimplemented; this is a blunt global `read()` branch.

Coherency passing C1–C5 means it is not silently returning wrong bytes in these scenarios, but it is **too slow and too memory-heavy** to promote.

---

## Artifacts

- A/B baseline: `2026-06-29T09-16-57_baseline_MmapRead_*`
- A/B experimental: `2026-06-29T09-56-26_experimental_MmapRead_*`
- Coherency: `2026-06-29T12-24-40_MmapRead_coherency.json`

Lab restored to `MmapRead: false` after coherency run.

No delete until you say so.
