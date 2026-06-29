# E1 — `UringFixedFile` decision packet (n=30, 2026-06-29)

> **Status: DELETED** (2026-06-29) — approved after n=30 lab run showed no io_read gain and TTI veto failure.

**Flag:** `UringFixedFile` (removed; was qvisor `experimental-uring-fixed-file`)  
**Lab:** `lab@lab.keska.vpn`  
**Iterations:** 30 per suite (baseline + experimental, same run)  
**Recommendation:** **Delete (veto failed)** — pending your explicit approval; **no code deleted**  
**Decision packet JSON:** `~/.keska-lab/results/group5/2026-06-29T07-14-01_UringFixedFile_decision_packet.json`

---

## A/B summary (p50, n=30)

| Metric | Baseline | Experimental | Delta | Gate |
|--------|----------|--------------|-------|------|
| light `tti_ms` | 52 ms | 59 ms | +13.5% | **FAIL** |
| light `tti_under_load_ms` | 45 ms | 48 ms | +6.7% | **FAIL** |
| light `memory_idle_rss_mb` | 20.77 MB | 20.73 MB | −0.2% | PASS |
| full `tti_ms` | 54 ms | 54 ms | 0% | — |
| full `io_read_mib_s` | 6041.6 | 6041.6 | 0% | **FAIL** (+10% bar) |
| full `io_write_mib_s` | 951.5 | 995.2 | +4.6% | FAIL |
| full `io_concurrent_read_mib_s` | 1915 | 2011 | +5.0% | PASS |
| db `pgbench_tps` | 322.6 | 337.2 | +4.5% | — |

Memory overhead negligible (~21 MB). The n=5 **+39% io_read** gain did **not** reproduce at n=30 (baseline already at ~6.2 GiB/s p50). Light-suite TTI regression is clearer with more samples (+7 ms p50).

Experimental db had 2 errors (1× SSH timeout, 1× script exit 1 on `tti_ms`); baseline full had 2 errors on `tti_ms` as well.

---

## Artifacts

- Baseline: `~/.keska-lab/results/group5/2026-06-29T06-40-00_baseline_UringFixedFile_quark_{light,full,db}.json`
- Experimental: `~/.keska-lab/results/group5/2026-06-29T07-14-01_experimental_UringFixedFile_quark_{light,full,db}.json`

Lab restored to baseline config after run.

---

## Prior runs

- n=5 post-fix (2026-06-28): showed +39% io_read but +8% light TTI — see `2026-06-28T23-30-41_*` artifacts.
- Pre-fix aborted run: invalid (deadlock / orphan leak) — see bugs 014/015.

No promote or delete until you say so.
