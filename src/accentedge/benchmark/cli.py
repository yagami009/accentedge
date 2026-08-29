"""CLI for AccentEdge BPO Benchmark v1."""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="AccentEdge BPO Benchmark v1 CLI")
console = Console()


@app.command()
def validate_dataset(
    manifest_path: str = typer.Option("data/manifests/benchmark.parquet", "--manifest"),
):
    """Validate dataset manifest and audio files."""
    console.print(f"Validating dataset: {manifest_path}")
    try:
        from ..dataset.validator import DatasetValidator
        validator = DatasetValidator(Path(manifest_path))
        result = validator.validate()
        if result["valid"]:
            console.print("[green]Dataset validation PASSED[/green]")
        else:
            console.print(f"[red]Dataset validation FAILED: {len(result['issues'])} issues[/red]")
            for issue in result["issues"]:
                console.print(f"  - {issue}")
            raise typer.Exit(1)
    except ImportError:
        console.print("[yellow]Dataset validator not yet implemented[/yellow]")


@app.command()
def build_splits(
    config: str = typer.Option("configs/splits.yaml", "--config"),
    output: str = typer.Option("data/manifests/splits.parquet", "--output"),
):
    """Build speaker-disjoint train/test splits."""
    console.print(f"Building splits from {config}")
    try:
        from ..dataset.splits import build_splits
        build_splits(Path(config), Path(output))
        console.print(f"[green]Splits written to {output}[/green]")
    except ImportError:
        console.print("[yellow]Split builder not yet implemented[/yellow]")


@app.command()
def run(
    candidate: str = typer.Option("passthrough", "--candidate"),
    split: str = typer.Option("dev", "--split"),
    condition: str = typer.Option("clean", "--condition"),
    output_dir: str = typer.Option("runs/", "--output-dir"),
    strength: float | None = typer.Option(None, "--strength"),
):
    """Run benchmark against a candidate."""
    console.print(f"Running benchmark: candidate={candidate}, split={split}, condition={condition}")
    try:
        from ..candidates.registry import get_adapter
        from ..runner.benchmark import BenchmarkRunner
        adapter = get_adapter(candidate)
        runner = BenchmarkRunner(
            candidate=adapter,
            split=split,
            condition=condition,
            output_dir=output_dir,
            conversion_strength=strength,
        )
        result = runner.run([])
        console.print(f"[green]Completed: {result['succeeded']}/{result['total_items']}[/green]")
        if result["failed"] > 0:
            console.print(f"[yellow]Failed: {result['failed']} items[/yellow]")
    except ImportError:
        console.print("[yellow]Benchmark runner not yet implemented[/yellow]")


@app.command()
def report(
    run_dir: str = typer.Argument("runs/latest"),
    format: str = typer.Option("html", "--format"),
):
    """Generate benchmark report."""
    console.print(f"Generating {format} report for {run_dir}")
    try:
        if format == "html":
            from ..reporting.html import generate_html_report
            generate_html_report({}, {}, [], output_path=f"{run_dir}/report.html")
        elif format == "json":
            from ..reporting.json_report import generate_json_report
            generate_json_report({}, f"{run_dir}/report.json")
        console.print(f"[green]Report written to {run_dir}/report.{format}[/green]")
    except ImportError:
        console.print("[yellow]Report generator not yet implemented[/yellow]")


def main():
    app()


if __name__ == "__main__":
    main()
