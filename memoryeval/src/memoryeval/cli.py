"""MemoryEval CLI — run benchmarks from the command line.

Entry point: ``memoryeval`` (configured in pyproject.toml).

Commands
--------
run          Run the full benchmark suite against a memory backend.
show-config  Validate and pretty-print a config file without running.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from memoryeval import __version__
from memoryeval.config import MemoryEvalConfig
from memoryeval.report import ReportRenderer
from memoryeval.scorer import BenchmarkScorer
from memoryeval.types import BenchmarkCategory, BenchmarkReport, CaseResult

# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

def _build_adapter(cfg: MemoryEvalConfig) -> Any:
    """Instantiate the correct adapter from config.

    Adapters for real backends (qdrant, chroma) are imported lazily so that
    memoryeval can be installed without those backend packages if only the
    in-memory adapter is needed.

    Raises:
        click.UsageError: if the backend name is unknown or the required
                          connection parameters are missing.
    """
    bc = cfg.backend
    name = bc.name.lower()

    if name == "memory":
        from engram.adapters.memory import InMemoryAdapter
        return InMemoryAdapter()

    if name == "qdrant":
        try:
            from engram.adapters.qdrant import QdrantAdapter
        except ImportError as exc:
            raise click.UsageError(
                "qdrant backend requires 'qdrant-client': pip install qdrant-client"
            ) from exc
        url = bc.connection.get("url")
        if not url:
            raise click.UsageError(
                "qdrant backend requires connection.url in the config file"
            )
        return QdrantAdapter(
            location=url,
            collection=bc.collection,
            vector_size=bc.vector_size,
            api_key=bc.connection.get("api_key"),
        )

    if name == "chroma":
        try:
            import chromadb

            from engram.adapters.chroma import ChromaAdapter
        except ImportError as exc:
            raise click.UsageError(
                "chroma backend requires 'chromadb': pip install chromadb"
            ) from exc
        host = bc.connection.get("host", "localhost")
        port = int(bc.connection.get("port", 8000))
        client = chromadb.HttpClient(host=host, port=port)
        return ChromaAdapter(
            client=client,
            collection=bc.collection,
            vector_size=bc.vector_size,
        )

    raise click.UsageError(
        f"Unknown backend {bc.name!r}. Supported: memory, qdrant, chroma"
    )


# ---------------------------------------------------------------------------
# Async runner helper
# ---------------------------------------------------------------------------

async def _run_benchmark(
    cfg: MemoryEvalConfig,
    *,
    categories: list[BenchmarkCategory] | None,
    timeout: float,
    console: Console,
    no_progress: bool,
) -> BenchmarkReport:
    """Core async logic: open adapter → run scorer → close adapter."""
    adapter = _build_adapter(cfg)

    try:
        # Adapters that require an explicit open() call (e.g. Qdrant, Chroma)
        # also implement __aenter__; InMemoryAdapter's open() is a no-op.
        if hasattr(adapter, "open"):
            await adapter.open()

        scorer = BenchmarkScorer(
            adapter,
            backend_name=cfg.backend.name,
            case_timeout=timeout,
        )

        if no_progress:
            report = await scorer.run(categories=categories)
        else:
            report = await _run_with_progress(scorer, categories=categories, console=console)
    finally:
        await adapter.close()

    return report


async def _run_with_progress(
    scorer: BenchmarkScorer,
    *,
    categories: list[BenchmarkCategory] | None,
    console: Console,
) -> BenchmarkReport:
    """Run the scorer with a Rich progress bar ticking after each case."""
    results: list[CaseResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[case_name]}[/dim]"),
        console=console,
        transient=True,
    ) as progress:
        # We can't know total upfront without filtering — run a dry-count first
        active_count = sum(
            1 for cls in scorer._cases
            if categories is None or cls.category in categories
        )
        task = progress.add_task(
            "Running benchmark…",
            total=active_count,
            case_name="",
        )

        async def on_complete(result: CaseResult, i: int, total: int) -> None:
            results.append(result)
            status = "✓" if result.passed else "✗"
            progress.update(
                task,
                advance=1,
                case_name=f"{status} {result.case_name}",
            )

        report = await scorer.run(categories=categories, on_case_complete=on_complete)

    return report


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version=__version__, prog_name="memoryeval")
def main() -> None:
    """MemoryEval — benchmark your AI memory backend's reliability."""


# ---------------------------------------------------------------------------
# `memoryeval run`
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a YAML config file.",
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    type=click.Choice(
        [c.value for c in BenchmarkCategory],
        case_sensitive=False,
    ),
    help="Run only this category (repeatable). Omit to run all five.",
)
@click.option(
    "--format",
    "output_format",
    default="rich",
    type=click.Choice(["rich", "plain", "json"], case_sensitive=False),
    show_default=True,
    help="Output format.",
)
@click.option(
    "--timeout",
    default=30.0,
    show_default=True,
    type=float,
    help="Per-case timeout in seconds.",
)
@click.option(
    "--no-color",
    "no_color",
    is_flag=True,
    default=False,
    help="Disable colour in Rich output (also honoured via NO_COLOR env var).",
)
@click.option(
    "--no-progress",
    "no_progress",
    is_flag=True,
    default=False,
    help="Disable the progress bar (useful for CI log capture).",
)
def run(
    config_path: Path,
    categories: tuple[str, ...],
    output_format: str,
    timeout: float,
    no_color: bool,
    no_progress: bool,
) -> None:
    """Run the full benchmark suite against a memory backend.

    Example config (qdrant)::

    \b
        backend:
          name: qdrant
          collection: memoryeval
          vector_size: 1536
          connection:
            url: http://localhost:6333

    Example config (in-memory, for quick validation)::

    \b
        backend:
          name: memory

    Exit codes: 0 = all cases passed, 1 = one or more cases failed or errored.
    """
    try:
        cfg = MemoryEvalConfig.from_yaml(config_path)
    except Exception as exc:
        raise click.UsageError(f"Failed to load config: {exc}") from exc

    active_categories: list[BenchmarkCategory] | None = (
        [BenchmarkCategory(c) for c in categories] if categories else None
    )

    console = Console(no_color=no_color or "NO_COLOR" in os.environ)

    try:
        report = asyncio.run(
            _run_benchmark(
                cfg,
                categories=active_categories,
                timeout=timeout,
                console=console,
                no_progress=no_progress,
            )
        )
    except click.UsageError:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    renderer = ReportRenderer(report)
    fmt = output_format.lower()

    if fmt == "rich":
        renderer.to_rich(console=console)
    elif fmt == "plain":
        click.echo(renderer.to_plain())
    else:  # json
        click.echo(renderer.to_json())

    # Exit 1 if any case failed so CI pipelines can gate on the result
    failed = sum(1 for r in report.all_results if not r.passed)
    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# `memoryeval show-config`
# ---------------------------------------------------------------------------

@main.command("show-config")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def show_config(config_path: Path) -> None:
    """Validate and pretty-print a config file.

    Exits with code 1 if the file is invalid.
    """
    try:
        cfg = MemoryEvalConfig.from_yaml(config_path)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    console = Console()
    console.print()
    console.rule("[bold]MemoryEval Config[/bold]")
    console.print()
    console.print(f"  [bold]Backend:[/bold]     {cfg.backend.name}")
    console.print(f"  [bold]Collection:[/bold]  {cfg.backend.collection}")
    console.print(f"  [bold]Vector size:[/bold] {cfg.backend.vector_size}")

    if cfg.backend.connection:
        _SECRET_KEYS = {"api_key", "password", "token", "secret"}
        redacted = {
            k: "***" if k in _SECRET_KEYS else v
            for k, v in cfg.backend.connection.items()
        }
        console.print(f"  [bold]Connection:[/bold]  {redacted}")

    cats = (
        ", ".join(c.value for c in cfg.categories)
        if cfg.categories else "all"
    )
    console.print(f"  [bold]Categories:[/bold]  {cats}")
    console.print()
    console.print("  [green]Config is valid.[/green]")
    console.print()
