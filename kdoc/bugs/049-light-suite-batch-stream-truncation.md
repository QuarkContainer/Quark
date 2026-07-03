# Light suite batch loses exec samples when streaming SSH

## Symptom

`lab.quark.bench("light", n=10)` completed tti through `cpu_loop_ms` but failed with `expected 10 exec_in_running_ms samples, got 1`.

## Cause

SSH line-streaming mode truncated captured stdout on long batch scripts, so `parse_metric_samples` saw incomplete output.

## Fix

Removed SSH streaming from the lab harness entirely. All remote commands capture stdout/stderr via `subprocess.run`; setup/bench code no longer accepts a `stream` flag.

## Files

- `lab/src/keska_lab/remote.py`
- `lab/src/keska_lab/backends/quark.py`
- `lab/src/keska_lab/harness/runner.py`
- (and other lab callers)

## Verify

```bash
cd lab && python -m pytest tests/test_harness_batch.py -q
lab.quark.bench("light", n=10, setup=False)  # all 8 metrics × 10 samples
```
