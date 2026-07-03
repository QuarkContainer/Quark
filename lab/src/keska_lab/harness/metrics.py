"""Shared remote measurement helpers for harness backends."""

from __future__ import annotations

# Seconds after start before sampling idle/paused RSS (stable on Quark at 0.25s).
MEMORY_RSS_SETTLE_SECS = 0.25


def rss_by_args_match_shell(id_var: str = "$ID") -> str:
    """Awk snippet: sum RSS (MB) for processes whose args contain sandbox ID."""
    return (
        f"ID_VAL={id_var}\n"
        f'rss=$(ps -eo rss,args | awk -v id="$ID_VAL" '
        "'index($0, id) {s+=$1} END {printf \"%.2f\", s/1024}')"
    )


def quark_rss_for_sandbox_shell(quark_list_cmd: str, id_var: str = "$ID") -> str:
    """
    Sum RSS (MB) for one Quark sandbox.

    Prefer ``quark list`` PID + shallow process tree; fall back to args match
    when the PID column is empty.
    """
    args_fallback = rss_by_args_match_shell(id_var)
    # quark_list_cmd is a trusted shell fragment from _quark_cmd(), not a single binary.
    return (
        f"ID_VAL={id_var}\n"
        f'__root_pid=$({quark_list_cmd} 2>/dev/null | awk -v id="$ID_VAL" '
        "'$1==id {print $2; exit}')\n"
        'if [ -n "${__root_pid:-}" ] && [ "$__root_pid" != "-1" ]; then\n'
        '  __pids="$__root_pid"\n'
        '  for __c in $(pgrep -P "$__root_pid" 2>/dev/null); do\n'
        '    __pids="$__pids $__c"\n'
        '    for __gc in $(pgrep -P "$__c" 2>/dev/null); do __pids="$__pids $__gc"; done\n'
        "  done\n"
        '  rss=$(ps -o rss= -p $__pids 2>/dev/null | awk \'{s+=$1} END {printf "%.2f", s/1024}\')\n'
        "else\n"
        f"  {args_fallback}\n"
        "fi"
    )


def parse_float_lines(stdout: str, *, expect: int | None = None) -> list[float]:
    """Parse one numeric metric per non-empty line from remote batch output."""
    values: list[float] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("ERR "):
            continue
        if line.startswith("SAMPLE "):
            parts = line.split()
            if len(parts) >= 3:
                values.append(float(parts[-1]))
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    if expect is not None and len(values) != expect:
        raise ValueError(f"expected {expect} samples, got {len(values)} from:\n{stdout!r}")
    return values


def parse_metric_samples(stdout: str) -> dict[str, list[float]]:
    """Parse ``METRIC name SAMPLE idx value`` lines from a combined suite batch."""
    indexed: dict[str, list[tuple[int, float]]] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("METRIC "):
            continue
        parts = line.split()
        if len(parts) < 5 or parts[2] != "SAMPLE":
            continue
        metric = parts[1]
        try:
            idx = int(parts[3])
            val = float(parts[4])
        except ValueError:
            continue
        indexed.setdefault(metric, []).append((idx, val))
    return {
        metric: [val for _, val in sorted(pairs, key=lambda p: p[0])]
        for metric, pairs in indexed.items()
    }


def parse_pause_resume_lines(stdout: str, *, expect: int | None = None) -> list[tuple[float, float, float]]:
    """Parse ``pause_ms resume_ms rss_mb`` triples from batch output."""
    triples: list[tuple[float, float, float]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("ERR "):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        triples.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if expect is not None and len(triples) != expect:
        raise ValueError(f"expected {expect} pause/resume triples, got {len(triples)}")
    return triples
