"""MemoryEval CLI — run benchmarks from the command line."""

import click


@click.group()
@click.version_option(package_name="memoryeval")
def main() -> None:
    """MemoryEval — benchmark your AI memory system's reliability."""


@main.command()
@click.option(
    "--backend",
    required=True,
    type=click.Choice(["qdrant", "chroma", "weaviate", "pgvector"], case_sensitive=False),
    help="Storage backend to benchmark against.",
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to backend connection config (YAML).",
)
def run(backend: str, config: str) -> None:
    """Run the full benchmark suite against a memory backend."""
    click.echo(f"Backend: {backend} | Config: {config}")
    click.echo("MemoryEval benchmark suite — coming in Step 3.")
