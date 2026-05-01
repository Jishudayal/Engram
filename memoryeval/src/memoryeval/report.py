"""MemoryEval report renderer.

Three output formats:

- ``to_rich(console)``  — coloured terminal table using Rich; default stdout.
- ``to_plain()``        — ASCII text string; no ANSI, safe for log files / CI.
- ``to_json(indent)``   — JSON string; machine-readable, suitable for CI artefacts.
"""

from __future__ import annotations

import json

from rich import box as rich_box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from memoryeval.types import BenchmarkReport, CaseResult, HallucinationRisk

__all__ = ["ReportRenderer"]

# ── colour maps ──────────────────────────────────────────────────────────────

_RISK_STYLE: dict[HallucinationRisk, str] = {
    HallucinationRisk.LOW:      "bold green",
    HallucinationRisk.MEDIUM:   "bold yellow",
    HallucinationRisk.HIGH:     "bold red",
    HallucinationRisk.CRITICAL: "bold bright_red",
}


def _score_style(score: float) -> str:
    if score >= 0.85:
        return "green"
    if score >= 0.70:
        return "yellow"
    if score >= 0.50:
        return "red"
    return "bright_red"


# ── renderer ─────────────────────────────────────────────────────────────────

class ReportRenderer:
    """Render a BenchmarkReport in Rich, plain-text, or JSON format.

    Usage::

        renderer = ReportRenderer(report)
        renderer.to_rich()          # prints to stdout with colour
        text = renderer.to_plain()  # returns ASCII string
        blob = renderer.to_json()   # returns JSON string
    """

    def __init__(self, report: BenchmarkReport) -> None:
        self._report = report

    # ── Rich ─────────────────────────────────────────────────────────────────

    def to_rich(self, *, console: Console | None = None) -> None:
        """Print the report to a Rich Console (default: stdout).

        Pass a Console built on an ``io.StringIO`` to capture the output
        in tests without printing to a terminal.
        """
        con = console or Console()
        r   = self._report

        run_at = r.run_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        con.print()
        con.rule(f"[bold]MemoryEval[/bold]  ·  [cyan]{r.backend_name}[/cyan]  ·  {run_at}")
        con.print()

        # ── category summary table ────────────────────────────────────────
        table = Table(
            box=rich_box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Category",  style="cyan",    min_width=16)
        table.add_column("Score",     justify="right", min_width=6)
        table.add_column("Pass Rate", justify="right", min_width=9)
        table.add_column("Status",    justify="center", min_width=6)

        for cs in r.categories:
            n      = len(cs.results)
            passed = sum(1 for res in cs.results if res.passed)
            all_ok = passed == n

            score_text  = Text(f"{cs.score:.3f}", style=_score_style(cs.score))
            pass_text   = Text(f"{passed}/{n}",   style="green" if all_ok else "yellow")
            status_text = Text("✓" if all_ok else "✗", style="green" if all_ok else "red")

            table.add_row(cs.category.value, score_text, pass_text, status_text)

        con.print(table)

        # ── overall score + risk ──────────────────────────────────────────
        risk_style  = _RISK_STYLE[r.hallucination_risk]
        score_str   = f"{r.overall_score:.3f}"
        risk_label  = f"{r.hallucination_risk.value.upper()} RISK"

        con.print(
            f"  [bold]Overall[/bold]  "
            f"[{_score_style(r.overall_score)}]{score_str}[/{_score_style(r.overall_score)}]"
            f"  →  "
            f"[{risk_style}]{risk_label}[/{risk_style}]"
        )
        con.print()

        # ── failures ─────────────────────────────────────────────────────
        failures: list[CaseResult] = [res for res in r.all_results if not res.passed]
        total = len(r.all_results)
        if total == 0:
            con.print("  [dim]No cases ran.[/dim]")
        elif not failures:
            con.print(f"  [green]✓ All {total} cases passed[/green]")
        else:
            con.print(f"  [bold red]✗ {len(failures)} case(s) failed[/bold red]")
            for f in failures:
                error_part = f"  [dim]{f.error}[/dim]" if f.error else ""
                con.print(
                    f"    [dim]{f.category.value}[/dim] / [bold]{f.case_name}[/bold]"
                    f"  score={f.score:.2f}{error_part}"
                )

        con.print()

    # ── plain text ───────────────────────────────────────────────────────────

    def to_plain(self) -> str:
        """Return the report as a plain ASCII string (no ANSI, no box-drawing).

        Suitable for log files, CI output, and ``--format plain`` mode.
        """
        r     = self._report
        lines: list[str] = []
        W     = 60

        lines.append("MemoryEval Report")
        lines.append("=" * W)
        lines.append(f"Backend:    {r.backend_name}")
        lines.append(f"Run at:     {r.run_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"Report ID:  {r.report_id}")
        lines.append("")

        # table header
        col_cat   = 20
        col_score =  7
        col_pass  = 10
        hdr = (
            f"{'Category':<{col_cat}}"
            f"{'Score':>{col_score}}"
            f"{'Pass Rate':>{col_pass}}"
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))

        for cs in r.categories:
            n      = len(cs.results)
            passed = sum(1 for res in cs.results if res.passed)
            fraction = f"{passed}/{n}"
            lines.append(
                f"{cs.category.value:<{col_cat}}"
                f"{cs.score:>{col_score}.3f}"
                f"{fraction:>{col_pass}}"
            )

        lines.append("")
        lines.append(
            f"Overall Score:  {r.overall_score:.3f}  "
            f"[{r.hallucination_risk.value.upper()} RISK]"
        )
        lines.append("")

        failures: list[CaseResult] = [res for res in r.all_results if not res.passed]
        total = len(r.all_results)
        if total == 0:
            lines.append("No cases ran.")
        elif not failures:
            lines.append(f"All {total} cases passed.")
        else:
            lines.append(f"{len(failures)} case(s) failed:")
            for f in failures:
                error_part = f"  error={f.error!r}" if f.error else ""
                lines.append(
                    f"  [{f.category.value}] {f.case_name}"
                    f"  score={f.score:.2f}{error_part}"
                )

        return "\n".join(lines) + "\n"

    # ── JSON ─────────────────────────────────────────────────────────────────

    def to_json(self, *, indent: int = 2) -> str:
        """Return the report serialised as a JSON string.

        Only the fields relevant for downstream tooling are included.
        The full Pydantic model is not dumped verbatim to keep the schema stable.
        """
        r = self._report
        data: dict[str, object] = {
            "schema_version":     "1",
            "report_id":          r.report_id,
            "run_at":             r.run_at.isoformat(),
            "backend_name":       r.backend_name,
            "overall_score":      r.overall_score,
            "hallucination_risk": r.hallucination_risk.value,
            "categories": [
                {
                    "category":  cs.category.value,
                    "score":     cs.score,
                    "pass_rate": cs.pass_rate,
                    "results": [
                        {
                            "case_name":      res.case_name,
                            "score":          res.score,
                            "passed":         res.passed,
                            "pass_threshold": res.pass_threshold,
                            "details":        res.details,
                            "error":          res.error,
                        }
                        for res in cs.results
                    ],
                }
                for cs in r.categories
            ],
        }
        return json.dumps(data, indent=indent)
