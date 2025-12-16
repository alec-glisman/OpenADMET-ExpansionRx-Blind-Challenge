Leaderboard Analysis
====================

The ``admet.leaderboard`` package provides tools for scraping, analyzing, and visualizing leaderboard data from the OpenADMET ExpansionRx Challenge on HuggingFace Spaces.

Overview
--------

This package enables:

* Automated scraping of leaderboard data from Gradio-based HuggingFace Spaces
* Ranking analysis and performance metrics extraction
* Publication-quality visualization generation
* Comprehensive markdown and CSV report generation

Quick Start
-----------

CLI Usage
~~~~~~~~~

The easiest way to use the leaderboard tools is through the command-line interface:

.. code-block:: bash

   # Basic usage - scrape and analyze for a specific user
   admet leaderboard scrape --user your_username

   # Custom output directory
   admet leaderboard scrape --user your_username --output ./my_results

   # Skip plot generation (faster)
   admet leaderboard scrape --user your_username --no-plots

   # Use a different HuggingFace Space
   admet leaderboard scrape --user your_username --space owner/space-name

Python API Usage
~~~~~~~~~~~~~~~~

You can also use the leaderboard tools programmatically:

.. code-block:: python

   from admet.leaderboard import LeaderboardClient, LeaderboardConfig
   from admet.leaderboard.report import ResultsData, generate_markdown_report
   from admet.plot.leaderboard import generate_all_plots
   from pathlib import Path

   # Configure
   config = LeaderboardConfig(
       space="openadmet/OpenADMET-ExpansionRx-Challenge",
       target_user="your_username",
       cache_dir=Path("./results")
   )

   # Fetch data
   client = LeaderboardClient(config)
   tables = client.fetch_all_tables()

   # Analyze and generate reports
   # (see detailed examples below)

Configuration
-------------

The ``LeaderboardConfig`` dataclass controls scraping behavior:

.. code-block:: python

   from admet.leaderboard import LeaderboardConfig
   from pathlib import Path

   config = LeaderboardConfig(
       space="openadmet/OpenADMET-ExpansionRx-Challenge",  # HF Space
       target_user="your_username",                         # User to analyze
       cache_dir=Path("assets/submissions"),                # Output directory
       endpoints=[                                          # Task endpoints
           "LogD",
           "KSOL",
           "MLM CLint",
           "HLM CLint",
           "Caco-2 Permeability Efflux",
           "Caco-2 Permeability Papp A>B",
           "MPPB",
           "MBPB",
           "MGMB",
       ]
   )

Fetching Leaderboard Data
--------------------------

The ``LeaderboardClient`` handles API communication:

.. code-block:: python

   from admet.leaderboard import LeaderboardClient, LeaderboardConfig

   config = LeaderboardConfig(target_user="your_username")
   client = LeaderboardClient(config)

   # Fetch all tables (Average + all endpoints)
   tables = client.fetch_all_tables()

   # Access specific tables
   overall_df = tables["Average"]
   logd_df = tables["LogD"]

   # Fetch with retry configuration
   tables = client.fetch_all_tables(
       max_retries=5,      # Maximum retry attempts
       retry_delay=3.0     # Delay between retries (seconds)
   )

   # Clean up
   client.close()

Parsing and Analysis
--------------------

Parser utilities extract information from leaderboard tables:

.. code-block:: python

   from admet.leaderboard import find_user_rank, extract_value_uncertainty

   # Find user's rank in a table
   rank = find_user_rank(overall_df, "your_username")
   print(f"Your rank: {rank}")

   # Parse MAE values with uncertainty
   mae_str = "0.35 +/- 0.01"
   value, uncertainty = extract_value_uncertainty(mae_str)
   print(f"MAE: {value} ± {uncertainty}")

Generating Reports
------------------

Create comprehensive markdown reports:

.. code-block:: python

   from admet.leaderboard.report import (
       ResultsData,
       generate_markdown_report,
       save_csv_data,
       save_summary_statistics,
   )
   from pathlib import Path

   # Prepare results data
   results = ResultsData(
       summary_df=summary_df,          # Summary DataFrame
       tables=tables,                   # Raw tables
       task_mins=task_minimums,         # Min MAE per task
       overall_min=overall_minimum,     # Min overall MA-RAE
       target_user="your_username",
       timestamp="2025-12-16 10:00:00+00:00"
   )

   output_dir = Path("./results/2025-12-16")

   # Generate markdown report
   generate_markdown_report(
       results,
       output_dir / "report.md",
       include_figures=True
   )

   # Save CSV data
   save_csv_data(results, output_dir / "data")

   # Save summary statistics
   save_summary_statistics(results, output_dir / "summary.txt")

Visualization
-------------

Generate publication-quality plots:

.. code-block:: python

   from admet.plot.leaderboard import (
       generate_all_plots,
       plot_overall_rank_distribution,
       plot_task_rankings_bar,
       plot_delta_mae_comparison,
       save_figure_formats,
   )
   from pathlib import Path

   # Generate all plots at once
   generate_all_plots(
       results_df=summary_df,
       tables=tables,
       task_mins=task_minimums,
       overall_min=overall_minimum,
       output_dir=Path("./figures"),
       target_user="your_username"
   )

   # Or generate individual plots
   fig, axs = plot_overall_rank_distribution(
       tables["Average"],
       user_rank=5
   )
   save_figure_formats(fig, Path("./figures/overall_rank"))

   fig, ax = plot_task_rankings_bar(
       task_data,
       target_user="your_username"
   )
   save_figure_formats(fig, Path("./figures/task_rankings"))

Available Plots
~~~~~~~~~~~~~~~

The package generates 10+ plot types:

1. **Overall Rank Distribution** - Histogram and ECDF showing rank spread
2. **Task Rankings Bar Chart** - Horizontal bars with performance zones
3. **Delta MAE Comparison** - Performance gap vs. top performer
4. **MAE Comparison** - Side-by-side MAE with error bars
5. **Performance Category Pie** - Distribution across performance tiers
6. **Metrics Heatmap** - Multi-metric heatmaps (R², Spearman, Kendall, MAE)
7. **Rank vs R² Scatter** - Correlation between rank and R²
8. **Rank vs MAE Scatter** - Correlation between rank and MAE
9. **Rank vs Spearman Scatter** - Correlation between rank and Spearman R
10. **Rank vs Kendall Scatter** - Correlation between rank and Kendall's τ

Output Structure
----------------

Running the CLI produces this directory structure:

.. code-block:: text

   assets/submissions/2025-12-16/
   ├── report.md                    # Main markdown report
   ├── summary.txt                  # Summary statistics
   ├── data/
   │   ├── summary.csv              # Summary table
   │   ├── Average.csv              # Overall leaderboard
   │   ├── LogD.csv                 # Task-specific tables
   │   └── ...
   └── figures/
       ├── png/                     # PNG format (web)
       │   ├── 01_overall_rank_hist_ecdf.png
       │   ├── 02_task_rankings_bar.png
       │   └── ...
       ├── svg/                     # SVG format (vector)
       └── pdf/                     # PDF format (publication)

Examples
--------

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from admet.leaderboard import LeaderboardClient, LeaderboardConfig, find_user_rank
   from admet.leaderboard.report import ResultsData, generate_markdown_report
   from admet.plot.leaderboard import generate_all_plots
   from pathlib import Path
   from datetime import datetime
   import pandas as pd

   # 1. Configure
   config = LeaderboardConfig(target_user="your_username")
   output_dir = config.get_output_dir(datetime.now().strftime("%Y-%m-%d"))
   output_dir.mkdir(parents=True, exist_ok=True)

   # 2. Fetch data
   client = LeaderboardClient(config)
   tables = client.fetch_all_tables()

   # 3. Analyze
   results_rows = []
   task_mins = {}
   overall_min = None

   for endpoint in config.all_endpoints:
       df = tables[endpoint]
       rank = find_user_rank(df, config.target_user)

       if rank:
           user_row = df.iloc[rank - 1]
           # Extract metrics and build results_rows...

   results_df = pd.DataFrame(results_rows)

   # 4. Create results object
   results = ResultsData(
       summary_df=results_df,
       tables=tables,
       task_mins=task_mins,
       overall_min=overall_min,
       target_user=config.target_user,
       timestamp=datetime.now().isoformat()
   )

   # 5. Generate outputs
   generate_markdown_report(results, output_dir / "report.md")
   generate_all_plots(
       results_df, tables, task_mins, overall_min,
       output_dir / "figures", config.target_user
   )

Troubleshooting
---------------

Connection Issues
~~~~~~~~~~~~~~~~~

If you encounter connection errors:

.. code-block:: python

   # Increase retries and delay
   tables = client.fetch_all_tables(max_retries=5, retry_delay=5.0)

User Not Found
~~~~~~~~~~~~~~

If your username isn't found:

* Check spelling (case-insensitive)
* Ensure you have submissions on the leaderboard
* Try the markdown link format if your username contains special characters

Performance
~~~~~~~~~~~

Plot generation can be slow. To speed up:

.. code-block:: bash

   # Skip plots during initial testing
   admet leaderboard scrape --user your_username --no-plots

API Reference
-------------

See :doc:`/api/leaderboard` for complete API documentation.
