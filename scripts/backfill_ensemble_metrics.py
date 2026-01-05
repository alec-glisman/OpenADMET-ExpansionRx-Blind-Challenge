#!/usr/bin/env python
"""Retroactively add aggregated ensemble metrics to parent MLflow runs."""
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:8084"
EXPERIMENT_ID = "6"


def should_aggregate_metric(metric_name: str) -> bool:
    """Filter metrics for aggregation (exclude profiling, system, counters)."""
    if metric_name.startswith("profiling"):
        return False
    if metric_name.startswith("system/"):
        return False
    if metric_name in ("epoch", "step", "global_step"):
        return False
    return True


def aggregate_child_metrics(client: MlflowClient, parent_run_id: str, child_run_ids: list) -> dict:
    """Fetch and aggregate metrics from child runs."""
    all_metrics = {}

    for run_id in child_run_ids:
        try:
            run = client.get_run(run_id)
            if run.info.status != "FINISHED":
                continue
            for metric_name, value in run.data.metrics.items():
                if should_aggregate_metric(metric_name):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        except Exception as e:
            print(f"  Warning: Failed to fetch metrics from child run {run_id[:8]}: {e}")
            continue

    # Compute aggregates
    aggregated = {}
    for metric_name, values in all_metrics.items():
        if not values:
            continue
        mean_val = float(np.mean(values))
        aggregated[f"ensemble/{metric_name}_mean"] = mean_val
        if len(values) > 1:
            std_val = float(np.std(values, ddof=1))
            aggregated[f"ensemble/{metric_name}_stddev"] = std_val
        else:
            aggregated[f"ensemble/{metric_name}_stddev"] = 0.0

    return aggregated


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Get all runs
    runs = client.search_runs(experiment_ids=[EXPERIMENT_ID], max_results=500)

    # Find unique parent IDs and their children
    parent_children = {}
    for run in runs:
        parent_id = run.data.tags.get("mlflow.parentRunId")
        if parent_id:
            if parent_id not in parent_children:
                parent_children[parent_id] = []
            parent_children[parent_id].append(run.info.run_id)

    print(f"Found {len(parent_children)} parent runs to process")

    for parent_id, child_ids in parent_children.items():
        try:
            parent_run = client.get_run(parent_id)
            run_name = parent_run.info.run_name

            # Check if already has aggregated metrics (expect ~180-190 for complete runs)
            agg_count = sum(1 for k in parent_run.data.metrics.keys() if k.startswith("ensemble/"))
            if agg_count >= 100:  # Consider complete if 100+ metrics
                print(f"Skipping {run_name} ({parent_id[:8]}): already has {agg_count} aggregated metrics")
                continue

            # Only process FINISHED runs
            if parent_run.info.status != "FINISHED":
                print(f"Skipping {run_name} ({parent_id[:8]}): status is {parent_run.info.status}")
                continue

            print(f"Processing {run_name} ({parent_id[:8]}) with {len(child_ids)} children...")

            # Aggregate metrics
            aggregated = aggregate_child_metrics(client, parent_id, child_ids)

            if aggregated:
                # Log metrics to parent run
                for metric_name, value in aggregated.items():
                    client.log_metric(parent_id, metric_name, value)
                print(f"  Logged {len(aggregated)} aggregated metrics")
            else:
                print(f"  No metrics to aggregate")

        except Exception as e:
            print(f"Error processing {parent_id[:8]}: {e}")


if __name__ == "__main__":
    main()
