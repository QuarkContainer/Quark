# E2 — `MmapRead` decision packet (n=30, 2026-06-29)

**Flag:** `MmapRead` (qkernel `experimental-mmap-read`)  
**Lab:** `lab@lab.keska.vpn`  
**Recommendation:** **Delete (veto failed + implement failed)** — pending your explicit approval  
**Decision packet JSON:** `~/.keska-lab/results/group5/2026-06-29T09-56-26_MmapRead_decision_packet.json`

---

## A/B summary (p50, n=30 — same run, baseline + experimental)

| Metric | Baseline p50 | Experimental p50 | Delta | Gate |
|--------|--------------|------------------|-------|------|
| light `tti_ms` | 53 ms | 59 ms | +11.3% | **FAIL** |
| light `tti_under_load_ms` | 48 ms | 47 ms | −2.1% | PASS |
| light `memory_idle_rss_mb` | 20.86 MB | 20.92 MB | +0.3% | PASS |
| db `memory_idle_rss_mb` | 51.86 MB | 52.05 MB | +0.4% | **FAIL** (3 db `tti_ms` errors) |
| full **`io_read_mib_s`** | **6144** | **1843** | **−70%** | **FAIL** |
| db **`pgbench_tps`** | **328.9** | **319.2** | **−2.9%** | **FAIL** |
| full `io_write_mib_s` | 1010 | 929 | −8.1% | — |
| full `io_concurrent_read` | 1936 | 1401 | −27.6% | — |

### n=5 vs n=30 — not noise

| Metric | n=5 experimental p50 | n=30 experimental p50 |
|--------|----------------------|------------------------|
| `io_read_mib_s` | 1843 | **1843** (identical) |
| `pgbench_tps` | 290 | 319 |

Experimental **io_read** samples cluster tightly around **1843 MiB/s** (many exact repeats) vs baseline **4.6–6.2 GiB/s** — looks like a **systematic ceiling** in the mmap read path, not run-to-run variance.

**Veto:** FAIL (light TTI + db errors)  
**Implement:** FAIL (large io_read regression; pgbench flat/slightly down)

Coherency C1–C7 and 30-min postgres soak were not run in harness.

---

## Artifacts

- Baseline: `~/.keska-lab/results/group5/2026-06-29T09-16-57_baseline_MmapRead_quark_{light,full,db}.json`
- Experimental: `~/.keska-lab/results/group5/2026-06-29T09-56-26_experimental_MmapRead_quark_{light,full,db}.json`

Prior n=5 run: `2026-06-29T08-29-42_*` (same directional result).

Lab restored to `MmapRead: false`.

No promote or delete until you say so.
