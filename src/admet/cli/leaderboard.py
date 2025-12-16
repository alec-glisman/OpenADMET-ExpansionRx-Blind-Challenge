"""CLI commands for leaderboard scraping and analysis."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from admet.leaderboard import LeaderboardClient, LeaderboardConfig, find_user_rank
from admet.leaderboard.report import (
    ResultsData,
    generate_markdown_report,
    save_csv_data,
    save_summary_statistics,
)
from admet.plot.leaderboard import generate_all_plots
from admet.util.logging import configure_logging

logger = logging.getLogger(__name__)
console = Console()

leaderboard_app = typer.Typer(
    name="leaderboard",
    help="Scrape and analyze leaderboard data",
    no_args_is_help=True,
)


@leaderboard_app.command("scrape")
def scrape_command(
    user: str = typer.Option(..., "--user", "-u", help="Target username to analyze"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory (default: assets/submissions/<timestamp>)"
    ),
    space: str = typer.Option(
        "openadmet/OpenADMET-ExpansionRx-Challenge",
        "--space",
        "-s",
        help="HuggingFace Space identifier",
    ),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip plot generation"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
) -> None:
    """Scrape leaderboard, analyze results, and generate reports.

    Examples:

        admet leaderboard scrape --user aglisman

        admet leaderboard scrape --user myname --output ./results --no-plots
    """
    configure_logging(level=log_level)

    # Setup configuration
    timestamp = datetime.now().strftime("%Y-%m-%d")
    config = LeaderboardConfig(
        space=space,
        target_user=user,
        cache_dir=output if output else Path("assets/submissions"),
    )

    output_dir = config.get_output_dir(timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]OpenADMET Leaderboard Scraper[/bold blue]")
    console.print(f"User: [bold]{user}[/bold]")
    console.print(f"Space: {space}")
    console.print(f"Output: {output_dir}\n")

    # Fetch leaderboard data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching leaderboard data...", total=None)

        try:
            client = LeaderboardClient(config)
            tables = client.fetch_all_tables()
            progress.update(task, description="[green]✓ Fetched leaderboard data")
        except Exception as e:
            console.print(f"[bold red]Error fetching data: {e}[/bold red]")
            raise typer.Exit(code=1) from e

    # Save raw tables
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for endpoint, df in tables.items():
        safe_name = endpoint.replace(" ", "_").replace("/", "_")
        csv_path = data_dir / f"{safe_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Cached %s leaderboard to %s", endpoint, csv_path)

    # Analyze results
    console.print("\n[bold]Analyzing results...[/bold]")
    results_rows = []
    task_mins: dict[str, float] = {}
    overall_min: Optional[float] = None

    for endpoint in config.all_endpoints:
        if endpoint not in tables:
            continue

        df = tables[endpoint]
        user_rank = find_user_rank(df, user)

        if user_rank is None:
            console.print(f"[yellow]Warning: User '{user}' not found in {endpoint}[/yellow]")
            continue

        # Extract metrics from user's row
        user_row = df.iloc[user_rank - 1]  # Convert to 0-indexed

        row_data = {
            "task": "OVERALL" if endpoint == "Average" else endpoint,
            "rank": user_rank,
            "n_rows": len(df),
        }

        # Extract metrics
        for col in df.columns:
            col_lower = str(col).lower()
            if "mae" in col_lower or "ma-rae" in col_lower:
                row_data["mae" if endpoint != "Average" else "ma-rae"] = user_row[col]
            elif "r2" in col_lower or "r²" in col_lower:
                row_data["r2"] = user_row[col]
            elif "spearman" in col_lower:
                row_data["spearman r"] = user_row[col]
            elif "kendall" in col_lower:
                row_data["kendall's tau"] = user_row[col]

        results_rows.append(row_data)

        # Extract minimum values from rank #1
        if len(df) > 0:
            top_row = df.iloc[0]
            for col in df.columns:
                col_lower = str(col).lower()
                if "mae" in col_lower or "ma-rae" in col_lower:
                    from admet.leaderboard.parser import extract_value_uncertainty

                    val, _ = extract_value_uncertainty(top_row[col])
                    if val is not None:
                        if endpoint == "Average":
                            overall_min = val
                        else:
                            task_mins[endpoint] = val

    results_df = pd.DataFrame(results_rows)
    timestamp_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S+00:00")

    results = ResultsData(
        summary_df=results_df,
        tables=tables,
        task_mins=task_mins,
        overall_min=overall_min,
        target_user=user,
        timestamp=timestamp_iso,
    )

    # Print summary table
    console.print("\n[bold]Results Summary:[/bold]")
    console.print(results_df.to_string(index=False))

    # Generate reports
    console.print("\n[bold]Generating reports...[/bold]")
    report_path = output_dir / "report.md"
    generate_markdown_report(results, report_path, include_figures=not no_plots)
    console.print(f"[green]✓[/green] Markdown report: {report_path}")

    save_csv_data(results, data_dir)
    console.print(f"[green]✓[/green] CSV data: {data_dir}")

    summary_path = output_dir / "summary.txt"
    save_summary_statistics(results, summary_path)
    console.print(f"[green]✓[/green] Summary statistics: {summary_path}")

    # Generate plots
    if not no_plots:
        console.print("\n[bold]Generating plots...[/bold]")
        figures_dir = output_dir / "figures"

        try:
            generate_all_plots(
                results_df,
                tables,
                task_mins,
                overall_min,
                figures_dir,
                user,
            )
            console.print(f"[green]✓[/green] Plots saved to: {figures_dir}")
        except Exception as e:
            console.print(f"[yellow]Warning: Plot generation failed: {e}[/yellow]")
            logger.exception("Plot generation failed")

    console.print(f"\n[bold green]✓ Complete![/bold green] Results saved to: {output_dir}")


@leaderboard_app.command("report")
def report_command(
    data_dir: Path = typer.Argument(..., help="Directory containing cached leaderboard data"),
    user: str = typer.Option(..., "--user", "-u", help="Target username"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output report path"),
) -> None:
    """Generate report from cached leaderboard data.

    Examples:

        admet leaderboard report assets/submissions/2025-12-16/data --user aglisman
    """
    if not data_dir.exists():
        console.print(f"[bold red]Error: Data directory not found: {data_dir}[/bold red]")
        raise typer.Exit(code=1)

    console.print("[bold blue]Generating report from cached data[/bold blue]")
    console.print(f"Data directory: {data_dir}")
    console.print(f"User: {user}\n")

    # Load cached tables
    tables = {}
    for csv_file in data_dir.glob("*.csv"):
        if csv_file.name == "summary.csv":
            continue
        endpoint = csv_file.stem.replace("_", " ")
        tables[endpoint] = pd.read_csv(csv_file)

    if not tables:
        console.print("[bold red]Error: No CSV files found in data directory[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"Loaded {len(tables)} tables")

    # TODO: Implement full report regeneration logic
    console.print("[yellow]Full report regeneration not yet implemented[/yellow]")


if __name__ == "__main__":
    leaderboard_app()
