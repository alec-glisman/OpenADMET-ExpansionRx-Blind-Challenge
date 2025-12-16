import numpy as np
import pandas as pd

data_path = (
    "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0/train.csv"
)
target_cols = [
    "LogD",
    "Log KSOL",
    "Log HLM CLint",
    "Log MLM CLint",
    "Log Caco-2 Permeability Papp A>B",
    "Log Caco-2 Permeability Efflux",
    "Log MPPB",
    "Log MBPB",
    "Log MGMB",
]

try:
    df = pd.read_csv(data_path)
    print(f"Loaded data with {len(df)} rows.")

    weights = []
    counts = []
    print("\nCalculating Counts and Weights (Max(Count) / Count):")

    # First pass: get counts
    for col in target_cols:
        if col not in df.columns:
            counts.append(0)
            continue
        counts.append(df[col].count())

    max_count = max(counts) if counts else 0
    print(f"Maximum count across tasks: {max_count}")

    print(f"\n{'Task':<35} | {'Count':<6} | {'Linear':<8} | {'Clipped(10)':<11} | {'Sqrt':<8}")
    print("-" * 80)

    linear_weights = []
    clipped_weights = []
    sqrt_weights = []

    for i, col in enumerate(target_cols):
        count = counts[i]

        if count == 0:
            w_linear = 1.0
            w_clipped = 1.0
            w_sqrt = 1.0
        else:
            # Linear: N_max / N_t
            w_linear = max_count / count

            # Clipped: min(10, N_max / N_t)
            w_clipped = min(10.0, w_linear)

            # Sqrt: sqrt(N_max / N_t)
            w_sqrt = np.sqrt(w_linear)

        linear_weights.append(w_linear)
        clipped_weights.append(w_clipped)
        sqrt_weights.append(w_sqrt)

        print(f"{col:<35} | {count:<6} | {w_linear:<8.4f} | {w_clipped:<11.4f} | {w_sqrt:<8.4f}")

    print("\nRecommended: Clipped (10.0) to prevent instability from the rarest task.")
    print("Copy this list to your config (Clipped):")
    print("target_weights:")
    print("  type: choice")
    print(f"  values: [{list(np.round(clipped_weights, 4))}]")

except Exception as e:
    print(f"Error: {e}")
