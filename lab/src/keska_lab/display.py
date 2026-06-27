"""Rich terminal output for benchmarks and status."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from keska_lab.bench.report import BenchReport

console = Console()


def print_setup_report(report) -> None:
    table = Table(title=f"Setup: {report.pipeline}")
    table.add_column("Step")
    table.add_column("Status")
    table.add_column("Message")

    for step in report.steps:
        status = "[green]ok[/green]" if step.ok else "[red]fail[/red]"
        lines = [ln.strip() for ln in step.message.splitlines() if ln.strip()]
        msg = lines[-1] if lines else step.message
        if len(msg) > 120:
            msg = msg[:117] + "..."
        table.add_row(step.name, status, msg)

    console.print(table)
    console.print(f"Total setup time: {report.total_s:.1f}s")


def print_bench_report(report: BenchReport) -> None:
    table = Table(title=f"Benchmark: {report.backend} / {report.profile}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    s = report.stats
    table.add_row("Samples", str(s.samples))
    table.add_row("Success rate", f"{report.success_rate * 100:.1f}%")
    table.add_row("Mean (ms)", f"{s.mean:.2f}")
    table.add_row("P50 (ms)", f"{s.p50:.2f}")
    table.add_row("P95 (ms)", f"{s.p95:.2f}")
    table.add_row("P99 (ms)", f"{s.p99:.2f}")
    table.add_row("Min (ms)", f"{s.min:.2f}")
    table.add_row("Max (ms)", f"{s.max:.2f}")

    console.print(table)
    if report.notes:
        console.print(f"[dim]{report.notes}[/dim]")


def print_full_bench_report(report) -> None:
    """Alias for suite report display."""
    print_suite_report(report)


def print_suite_report(report) -> None:
    from keska_lab.harness.stats import MetricStats

    title = f"Suite {getattr(report, 'suite', 'full')}: {report.backend}"
    if getattr(report, "workload", None):
        title += f" / {report.workload}"
    title += f" ({report.iterations} samples)"

    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("P50", justify="right")
    table.add_column("Unit", justify="left")

    for name, val in report.metrics.items():
        if isinstance(val, MetricStats):
            table.add_row(name, f"{val.mean:.2f}", f"{val.p50:.2f}", val.unit)
        else:
            table.add_row(name, str(val), "", "")

    console.print(table)
    if report.skipped:
        console.print(f"[dim]skipped: {', '.join(report.skipped)}[/dim]")
    if report.notes:
        for note in report.notes:
            console.print(f"[dim]{note}[/dim]")
    if report.errors:
        console.print(f"[yellow]{len(report.errors)} errors[/yellow] (see JSON)")


def print_suite_compare(
    left,
    right,
    *,
    left_label: str = "quark",
    right_label: str = "kata",
) -> None:
    from keska_lab.harness.stats import MetricStats

    metrics = left.metrics if hasattr(left, "metrics") else {}
    console.print(f"\n[bold]Compare {left_label} vs {right_label} (p50)[/bold]")
    if right_label == "kata":
        console.print("[dim]Kata/Firecracker is the reference runtime; Quark ratios <1.0 mean Quark is faster.[/dim]")
    for name in metrics:
        lv = metrics.get(name)
        rv = right.metrics.get(name) if hasattr(right, "metrics") else None
        if isinstance(lv, MetricStats) and isinstance(rv, MetricStats):
            ratio = lv.p50 / rv.p50 if rv.p50 else float("inf")
            console.print(
                f"{name}: {left_label} {lv.p50} vs {right_label} {rv.p50} {lv.unit}  ({ratio:.2f}x)"
            )
