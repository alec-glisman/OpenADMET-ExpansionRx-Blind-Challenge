import numpy as np
import pandas as pd

from admet.data.split import (
    build_cluster_label_matrix,
    cluster_multilabel_stratified_kfold,
)


def make_synthetic_clustered_df():
    # 6 clusters (0..5), 3 molecules per cluster
    rows = []
    for cluster in range(6):
        quality = "high" if cluster < 3 else "low"
        for i in range(3):
            # create task values with NaNs depending on cluster id
            if cluster % 3 == 0:
                A = 1.0
                B = 2.0
            elif cluster % 3 == 1:
                A = 1.0
                B = np.nan
            else:
                A = np.nan
                B = 2.0
            rows.append({"cluster": int(cluster), "A": A, "B": B, "quality": quality})
    df = pd.DataFrame(rows)
    return df


def test_build_cluster_label_matrix_quality_appended():
    df = make_synthetic_clustered_df()
    cluster_ids, cluster_labels, cluster_sizes = build_cluster_label_matrix(
        df, cluster_col="cluster", task_cols=["A", "B"], quality_col="quality", add_cluster_size=False
    )

    # Expect 6 clusters and 2 tasks + 2 unique qualities => 4 columns
    assert cluster_ids.shape[0] == 6
    assert cluster_labels.shape == (6, 4)

    # Determine expected per-cluster labels
    # First two columns are task presence (A, B)
    # Next two columns are quality presence (high, low) in lexicographic order
    expected_task_quality = []
    for cid in range(6):
        # tasks
        if cid % 3 == 0:
            ta, tb = 1, 1
        elif cid % 3 == 1:
            ta, tb = 1, 0
        else:
            ta, tb = 0, 1
        # quality
        if cid < 3:
            q_high, q_low = 1, 0
        else:
            q_high, q_low = 0, 1
        expected_task_quality.append([ta, tb, q_high, q_low])

    # cluster_ids are sorted; verify each row matches expected
    for idx, cid in enumerate(sorted(cluster_ids)):
        assert (cluster_labels[idx, :] == np.array(expected_task_quality[cid])).all()


def test_cluster_multilabel_stratified_kfold_balances_quality():
    df = make_synthetic_clustered_df()

    folds, diag = cluster_multilabel_stratified_kfold(
        df,
        cluster_col="cluster",
        task_cols=["A", "B"],
        quality_col="quality",
        n_folds=2,
        diagnostics=True,
        add_cluster_size=False,
        random_state=42,
    )

    # We expect 2 folds and ensure `folds` is a list
    assert isinstance(folds, list)
    assert len(folds) == 2

    # Count the number of training clusters per quality in each fold
    train_quality_counts = []
    for fold in folds:
        train_clusters = fold.train_clusters
        # For each training cluster, sample the first row's quality as canonical
        train_qs = []
        for cid in train_clusters:
            q = df[df["cluster"] == cid]["quality"].iloc[0]
            train_qs.append(q)
        counts = {q: train_qs.count(q) for q in set(train_qs)}
        train_quality_counts.append(counts)

    # Because there are 3 high and 3 low clusters, with 2 folds, each fold's
    # training clusters should have counts differing by at most 1 for each
    # quality level.
    highs = [c.get("high", 0) for c in train_quality_counts]
    lows = [c.get("low", 0) for c in train_quality_counts]

    assert max(highs) - min(highs) <= 1
    assert max(lows) - min(lows) <= 1
