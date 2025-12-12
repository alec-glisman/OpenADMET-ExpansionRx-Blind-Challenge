#!/usr/bin/env python3
"""
Compute task affinity using legacy gradient cosine approach and save artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from admet.model.chemprop.task_affinity import (
    TaskAffinityConfig,
    compute_task_affinity,
    plot_task_affinity_clustermap,
    plot_task_affinity_heatmap,
)


def parse_targets(target_str: str) -> list[str]:
    return [t.strip() for t in target_str.split(",") if t.strip()]


def main():
    parser = argparse.ArgumentParser(description="Compute and save task affinity")
    parser.add_argument("csv", help="Path to CSV with SMILES and targets")
    parser.add_argument("--smiles", default="SMILES", help="SMILES column name")
    parser.add_argument("--targets", required=True, help="Comma-separated target column names")
    parser.add_argument("--outdir", default=".", help="Output directory for artifacts")
    parser.add_argument("--n_groups", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_plots", action="store_true")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    target_cols = parse_targets(args.targets)

    config = TaskAffinityConfig(
        enabled=True,
        n_groups=args.n_groups,
        affinity_epochs=args.epochs,
        affinity_batch_size=args.batch_size,
    )

    affinity, tasks, groups = compute_task_affinity(df, args.smiles, target_cols, config=config)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "affinity_matrix.csv"
    pd.DataFrame(affinity, index=tasks, columns=tasks).to_csv(csv_path)
    print(f"Saved affinity matrix to {csv_path}")

    if args.save_plots:
        heat_path = outdir / "affinity_heatmap.png"
        fig_hm = plot_task_affinity_heatmap(affinity, tasks, save_path=str(heat_path))
        fig_hm.clf()
        print(f"Saved heatmap to {heat_path}")

        clus_path = outdir / "affinity_clustermap.png"
        fig_cm = plot_task_affinity_clustermap(affinity, tasks, groups=groups, save_path=str(clus_path))
        fig_cm.clf()
        print(f"Saved clustermap to {clus_path}")


if __name__ == "__main__":
    main()
