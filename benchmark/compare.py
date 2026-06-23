#!/usr/bin/env python3
"""
Compare Quark benchmark JSON reports over time.

Usage:
  python3 benchmark/compare.py baseline.json candidate.json
  python3 benchmark/compare.py a.json b.json --json-delta delta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REGRESSION_THRESHOLD_PCT = 10.0  # flag if candidate is worse by this %


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != "1.0":
        print(f"warning: unexpected schema in {path}", file=sys.stderr)
    return data


def flatten_metrics(report: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    suites = report.get("suites", {})
    for suite_name, suite_data in suites.items():
        if not isinstance(suite_data, dict):
            continue
        base = f"{prefix}{suite_name}" if not prefix else f"{suite_name}"

        if "stats" in suite_data and isinstance(suite_data["stats"], dict):
            for k, v in suite_data["stats"].items():
                if isinstance(v, (int, float)):
                    out[f"{base}.{k}"] = float(v)

        for key, val in suite_data.items():
            if key in ("stats", "waves", "unit", "skipped", "error", "method", "runs"):
                continue
            if isinstance(val, dict) and "stats" in val:
                for k, v in val["stats"].items():
                    if isinstance(v, (int, float)):
                        out[f"{base}.{key}.{k}"] = float(v)
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                out[f"{base}.{key}"] = float(val)

        # nested rss_mb etc
        if "rss_mb" in suite_data and isinstance(suite_data["rss_mb"], dict):
            for k, v in suite_data["rss_mb"].items():
                if isinstance(v, (int, float)):
                    out[f"{base}.rss_mb.{k}"] = float(v)

        if "read_mib_s" in suite_data:
            out[f"{base}.read_mib_s"] = float(suite_data.get("read_mib_s", 0))
        if "write_mib_s" in suite_data:
            out[f"{base}.write_mib_s"] = float(suite_data.get("write_mib_s", 0))

    return out


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    b = flatten_metrics(baseline)
    c = flatten_metrics(candidate)
    keys = sorted(set(b) | set(c))
    rows: list[dict[str, Any]] = []

    # metrics where lower is better
    lower_better = {"ms", "mean", "p50", "p95", "p99", "max", "rss", "delta", "latency"}

    for key in keys:
        bv = b.get(key)
        cv = c.get(key)
        if bv is None and cv is None:
            continue
        delta_pct: float | None = None
        regression = False
        if bv is not None and cv is not None and bv != 0:
            delta_pct = round((cv - bv) / abs(bv) * 100, 2)
            worse_if_higher = not any(x in key.lower() for x in lower_better)
            if worse_if_higher:
                regression = delta_pct > REGRESSION_THRESHOLD_PCT
            else:
                regression = delta_pct > REGRESSION_THRESHOLD_PCT
        rows.append(
            {
                "metric": key,
                "baseline": bv,
                "candidate": cv,
                "delta_pct": delta_pct,
                "regression": regression,
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]], baseline_label: str, candidate_label: str) -> None:
    print(f"\n{'Metric':<45} {baseline_label:>12} {candidate_label:>12} {'Δ%':>8} {'':>4}")
    print("-" * 85)
    for r in rows:
        bv = r["baseline"]
        cv = r["candidate"]
        dp = r["delta_pct"]
        flag = " ⚠" if r["regression"] else ""
        bvs = f"{bv:.2f}" if bv is not None else "—"
        cvs = f"{cv:.2f}" if cv is not None else "—"
        dps = f"{dp:+.1f}" if dp is not None else "—"
        print(f"{r['metric']:<45} {bvs:>12} {cvs:>12} {dps:>8}{flag}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Quark benchmark reports")
    parser.add_argument("reports", nargs="+", type=Path, help="JSON reports (first=baseline)")
    parser.add_argument("--json-delta", type=Path, help="Write delta JSON to file")
    parser.add_argument("--threshold", type=float, default=REGRESSION_THRESHOLD_PCT)
    args = parser.parse_args(argv)

    if len(args.reports) < 2:
        print("Need at least two report files", file=sys.stderr)
        return 1

    baseline_path = args.reports[0]
    candidate_path = args.reports[1]
    baseline = load_report(baseline_path)
    candidate = load_report(candidate_path)

    print(f"Baseline:  {baseline_path.name}  "
          f"({baseline.get('metadata', {}).get('git_commit', '?')}, "
          f"{baseline.get('metadata', {}).get('timestamp', '?')})")
    print(f"Candidate: {candidate_path.name}  "
          f"({candidate.get('metadata', {}).get('git_commit', '?')}, "
          f"{candidate.get('metadata', {}).get('timestamp', '?')})")

    rows = compare_reports(baseline, candidate)
    print_table(rows, baseline_path.stem[:12], candidate_path.stem[:12])

    regressions = [r for r in rows if r["regression"]]
    if regressions:
        print(f"\n⚠ {len(regressions)} metric(s) regressed > {args.threshold}%")
    else:
        print("\n✓ No major regressions detected")

    if args.json_delta:
        delta = {
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
            "rows": rows,
        }
        args.json_delta.write_text(json.dumps(delta, indent=2) + "\n")
        print(f"Delta written: {args.json_delta}")

    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
