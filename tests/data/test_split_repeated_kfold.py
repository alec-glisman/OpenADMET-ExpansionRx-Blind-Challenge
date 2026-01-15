"""
Tests for repeated k-fold cross-validation with cluster integrity.

Verifies that:
1. Different splits (repeats) produce different fold assignments
2. Clusters are never split between train and validation sets
3. All molecules within a cluster have the same train/val assignment
4. Edge cases are handled correctly (single-molecule clusters, uneven distributions)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from admet.data.split import (
    cluster_group_kfold,
    cluster_kfold,
    cluster_multilabel_stratified_kfold,
    cluster_stratified_kfold,
    pipeline,
)

# ---------------------------------------------------------------------------
# Fixtures for synthetic clustered data
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_clustered_df() -> pd.DataFrame:
    """Create synthetic data with 20 clusters of varying sizes (3-7 molecules each).

    Uses 20 clusters to ensure enough samples per label class for StratifiedKFold
    and enough variation for GroupKFold randomization testing.
    """
    np.random.seed(42)
    rows = []
    # 20 clusters with varying sizes (larger dataset for stratification)
    cluster_sizes = [3, 5, 4, 7, 3, 6, 4, 5, 3, 4, 5, 6, 4, 3, 5, 4, 6, 3, 5, 4]

    for cluster_id, size in enumerate(cluster_sizes):
        quality = ["high", "medium", "low"][cluster_id % 3]
        for _ in range(size):
            # Create task values with some NaNs to simulate missing data
            # but ensure enough non-NaN values per cluster for stratification
            task_a = np.random.randn() if np.random.rand() > 0.1 else np.nan
            task_b = np.random.randn() if np.random.rand() > 0.15 else np.nan
            task_c = np.random.randn() if np.random.rand() > 0.1 else np.nan
            rows.append(
                {
                    "cluster": cluster_id,
                    "TaskA": task_a,
                    "TaskB": task_b,
                    "TaskC": task_c,
                    "Quality": quality,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def large_clustered_df() -> pd.DataFrame:
    """Create larger synthetic data with 50 clusters for statistical tests."""
    np.random.seed(123)
    rows = []

    for cluster_id in range(50):
        size = np.random.randint(2, 10)
        quality = ["high", "medium", "low"][cluster_id % 3]
        for _ in range(size):
            task_a = np.random.randn() if np.random.rand() > 0.15 else np.nan
            task_b = np.random.randn() if np.random.rand() > 0.25 else np.nan
            rows.append(
                {
                    "cluster": cluster_id,
                    "TaskA": task_a,
                    "TaskB": task_b,
                    "Quality": quality,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def single_molecule_clusters_df() -> pd.DataFrame:
    """Create data with many single-molecule clusters (edge case)."""
    rows = []
    # 15 single-molecule clusters
    for cluster_id in range(15):
        quality = ["high", "medium", "low"][cluster_id % 3]
        rows.append(
            {
                "cluster": cluster_id,
                "TaskA": np.random.randn(),
                "TaskB": np.random.randn(),
                "Quality": quality,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def uneven_clusters_df() -> pd.DataFrame:
    """Create data with highly uneven cluster sizes (1 large, many small)."""
    np.random.seed(456)
    rows = []

    # One very large cluster (30 molecules)
    for _ in range(30):
        rows.append(
            {
                "cluster": 0,
                "TaskA": np.random.randn(),
                "TaskB": np.random.randn(),
                "Quality": "high",
            }
        )

    # 14 small clusters (2-3 molecules each)
    for cluster_id in range(1, 15):
        size = np.random.randint(2, 4)
        quality = ["medium", "low"][cluster_id % 2]
        for _ in range(size):
            rows.append(
                {
                    "cluster": cluster_id,
                    "TaskA": np.random.randn(),
                    "TaskB": np.random.randn(),
                    "Quality": quality,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test: Different splits produce different fold assignments
# ---------------------------------------------------------------------------


class TestDifferentSplitsHaveDifferentFolds:
    """Verify that repeated splits (with different random seeds) produce different folds."""

    def test_multilabel_stratified_different_seeds_produce_different_folds(self, synthetic_clustered_df: pd.DataFrame):
        """Same data with different random_state should yield different fold assignments."""
        task_cols = ["TaskA", "TaskB", "TaskC"]

        folds_seed_42 = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        folds_seed_43 = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=43,
        )

        # At least one fold should have different validation clusters
        any_different = False
        for f1, f2 in zip(folds_seed_42, folds_seed_43):
            if not np.array_equal(sorted(f1.val_clusters), sorted(f2.val_clusters)):
                any_different = True
                break

        assert any_different, "Different random seeds should produce different fold assignments"

    def test_stratified_different_seeds_produce_different_folds(self, large_clustered_df: pd.DataFrame):
        """StratifiedKFold with different seeds should yield different assignments.

        Note: Uses large_clustered_df (50 clusters) to ensure enough samples per label class.
        StratifiedKFold requires at least n_folds samples in each stratified class.
        """
        task_cols = ["TaskA", "TaskB"]

        # Use only 3 folds to avoid "n_splits > members in each class" error
        folds_seed_42 = cluster_stratified_kfold(
            large_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=3,
            shuffle=True,
            random_state=42,
        )

        folds_seed_100 = cluster_stratified_kfold(
            large_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=3,
            shuffle=True,
            random_state=100,
        )

        any_different = False
        for f1, f2 in zip(folds_seed_42, folds_seed_100):
            if not np.array_equal(sorted(f1.val_clusters), sorted(f2.val_clusters)):
                any_different = True
                break

        assert any_different, "Different random seeds should produce different fold assignments"

    def test_group_kfold_deterministic_by_design(self, large_clustered_df: pd.DataFrame):
        """GroupKFold is deterministic by design - same fold assignments regardless of seed.

        sklearn's GroupKFold assigns groups to folds to balance fold sizes deterministically.
        The random_state in cluster_group_kfold only shuffles row order within groups,
        not the fold assignments themselves. This is expected behavior.
        """
        task_cols = ["TaskA", "TaskB"]

        folds_seed_42 = cluster_group_kfold(
            large_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            n_folds=5,
            random_state=42,
        )

        folds_seed_99 = cluster_group_kfold(
            large_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            n_folds=5,
            random_state=99,
        )

        # GroupKFold should produce the same fold assignments regardless of seed
        all_same = True
        for f1, f2 in zip(folds_seed_42, folds_seed_99):
            if not np.array_equal(sorted(f1.val_clusters), sorted(f2.val_clusters)):
                all_same = False
                break

        assert all_same, (
            "GroupKFold should be deterministic - same fold assignments regardless of seed. "
            "If this fails, the implementation may have changed to use randomization."
        )

    def test_same_seed_produces_reproducible_folds(self, synthetic_clustered_df: pd.DataFrame):
        """Same random_state should produce identical fold assignments."""
        task_cols = ["TaskA", "TaskB", "TaskC"]

        folds_run1 = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        folds_run2 = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        for f1, f2 in zip(folds_run1, folds_run2):
            assert np.array_equal(
                sorted(f1.val_clusters), sorted(f2.val_clusters)
            ), "Same seed should produce identical folds"
            assert np.array_equal(
                sorted(f1.train_clusters), sorted(f2.train_clusters)
            ), "Same seed should produce identical folds"


# ---------------------------------------------------------------------------
# Test: Clusters are never split between train and validation
# ---------------------------------------------------------------------------


class TestClusterIntegrity:
    """Verify that clusters are kept intact (never split across train/val)."""

    @pytest.mark.parametrize(
        "split_method",
        ["group_kfold", "multilabel_stratified_kfold"],
    )
    def test_no_cluster_appears_in_both_train_and_val(self, synthetic_clustered_df: pd.DataFrame, split_method: str):
        """No cluster should appear in both train and validation sets."""
        task_cols = ["TaskA", "TaskB", "TaskC"]

        folds = cluster_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            split_method=split_method,
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        for fold in folds:
            train_clusters = set(fold.train_clusters)
            val_clusters = set(fold.val_clusters)

            overlap = train_clusters & val_clusters
            assert len(overlap) == 0, (
                f"Fold {fold.fold_id}: Clusters {overlap} appear in both train and val. "
                f"This violates cluster integrity and causes data leakage."
            )

    def test_stratified_kfold_no_cluster_overlap(self, large_clustered_df: pd.DataFrame):
        """Test stratified_kfold with larger dataset to avoid label class issues."""
        task_cols = ["TaskA", "TaskB"]

        folds = cluster_stratified_kfold(
            large_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=3,  # Use fewer folds to avoid rare label class issues
            shuffle=True,
            random_state=42,
        )

        for fold in folds:
            train_clusters = set(fold.train_clusters)
            val_clusters = set(fold.val_clusters)

            overlap = train_clusters & val_clusters
            assert len(overlap) == 0, (
                f"Fold {fold.fold_id}: Clusters {overlap} appear in both train and val. "
                f"This violates cluster integrity and causes data leakage."
            )

    @pytest.mark.parametrize(
        "split_method",
        ["group_kfold", "multilabel_stratified_kfold"],
    )
    def test_all_clusters_are_assigned(self, synthetic_clustered_df: pd.DataFrame, split_method: str):
        """Every cluster should be assigned to either train or val in each fold."""
        task_cols = ["TaskA", "TaskB", "TaskC"]
        all_clusters = set(synthetic_clustered_df["cluster"].unique())

        folds = cluster_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            split_method=split_method,
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        for fold in folds:
            assigned_clusters = set(fold.train_clusters) | set(fold.val_clusters)
            assert assigned_clusters == all_clusters, (
                f"Fold {fold.fold_id}: Not all clusters assigned. " f"Missing: {all_clusters - assigned_clusters}"
            )

    @pytest.mark.parametrize(
        "split_method",
        ["group_kfold", "multilabel_stratified_kfold"],
    )
    def test_all_molecules_in_cluster_have_same_assignment(
        self, synthetic_clustered_df: pd.DataFrame, split_method: str
    ):
        """All molecules within a cluster must be in the same partition (train or val)."""
        task_cols = ["TaskA", "TaskB", "TaskC"]

        folds = cluster_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            split_method=split_method,
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        for fold in folds:
            train_set = set(fold.train_indices)
            val_set = set(fold.val_indices)

            # Group molecules by cluster and check assignments
            for cluster_id, group in synthetic_clustered_df.groupby("cluster"):
                mol_indices = set(group.index.tolist())
                in_train = mol_indices & train_set
                in_val = mol_indices & val_set

                # All molecules should be in exactly one partition
                assert len(in_train) == 0 or len(in_val) == 0, (
                    f"Fold {fold.fold_id}, Cluster {cluster_id}: "
                    f"Molecules split between train ({len(in_train)}) and val ({len(in_val)}). "
                    f"This indicates data leakage - entire clusters must stay together."
                )

                # All molecules should be assigned somewhere
                assert len(in_train) > 0 or len(in_val) > 0, (
                    f"Fold {fold.fold_id}, Cluster {cluster_id}: " f"No molecules assigned to train or val."
                )

    def test_train_val_indices_are_disjoint(self, synthetic_clustered_df: pd.DataFrame):
        """Train and validation indices must be completely disjoint."""
        task_cols = ["TaskA", "TaskB", "TaskC"]

        folds = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        for fold in folds:
            train_set = set(fold.train_indices)
            val_set = set(fold.val_indices)

            overlap = train_set & val_set
            assert len(overlap) == 0, (
                f"Fold {fold.fold_id}: {len(overlap)} molecule indices appear in both "
                f"train and val sets. This is a severe data leakage issue."
            )

    def test_all_molecules_assigned_exactly_once_per_fold(self, synthetic_clustered_df: pd.DataFrame):
        """Every molecule should appear in exactly one of train or val per fold."""
        task_cols = ["TaskA", "TaskB", "TaskC"]
        all_indices = set(synthetic_clustered_df.index.tolist())

        folds = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        for fold in folds:
            assigned = set(fold.train_indices) | set(fold.val_indices)
            assert assigned == all_indices, (
                f"Fold {fold.fold_id}: Not all molecules assigned. " f"Missing {len(all_indices - assigned)} molecules."
            )


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases for cluster-based splitting."""

    @pytest.mark.parametrize(
        "split_method",
        ["group_kfold", "multilabel_stratified_kfold"],
    )
    def test_single_molecule_clusters(self, single_molecule_clusters_df: pd.DataFrame, split_method: str):
        """Splitting should work correctly with single-molecule clusters."""
        task_cols = ["TaskA", "TaskB"]

        folds = cluster_kfold(
            single_molecule_clusters_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            split_method=split_method,
            n_folds=3,  # Fewer folds since we have 15 clusters
            shuffle=True,
            random_state=42,
        )

        assert len(folds) == 3

        for fold in folds:
            # Each cluster has exactly 1 molecule
            assert fold.n_train_clusters == fold.n_train_mols
            assert fold.n_val_clusters == fold.n_val_mols

            # No overlap between train and val
            train_clusters = set(fold.train_clusters)
            val_clusters = set(fold.val_clusters)
            assert len(train_clusters & val_clusters) == 0

    @pytest.mark.parametrize(
        "split_method",
        ["group_kfold", "multilabel_stratified_kfold"],
    )
    def test_uneven_cluster_distribution(self, uneven_clusters_df: pd.DataFrame, split_method: str):
        """Splitting should handle highly uneven cluster sizes."""
        task_cols = ["TaskA", "TaskB"]

        folds = cluster_kfold(
            uneven_clusters_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            split_method=split_method,
            n_folds=3,
            shuffle=True,
            random_state=42,
        )

        assert len(folds) == 3

        # Verify cluster integrity despite uneven sizes
        for fold in folds:
            train_clusters = set(fold.train_clusters)
            val_clusters = set(fold.val_clusters)
            assert len(train_clusters & val_clusters) == 0

            # All molecules in each cluster should be together
            for cluster_id, group in uneven_clusters_df.groupby("cluster"):
                mol_indices = set(group.index.tolist())
                in_train = mol_indices & set(fold.train_indices)
                in_val = mol_indices & set(fold.val_indices)
                assert len(in_train) == 0 or len(in_val) == 0

    def test_minimum_clusters_for_folds(self):
        """Test with exactly n_folds clusters (minimum viable case)."""
        # Create exactly 5 clusters for 5 folds
        rows = []
        for cluster_id in range(5):
            for _ in range(3):
                rows.append(
                    {
                        "cluster": cluster_id,
                        "TaskA": np.random.randn(),
                        "Quality": "high",
                    }
                )
        df = pd.DataFrame(rows)

        folds = cluster_multilabel_stratified_kfold(
            df,
            cluster_col="cluster",
            task_cols=["TaskA"],
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        assert len(folds) == 5

        # Each fold should have exactly 1 validation cluster (with 5 clusters, 5 folds)
        for fold in folds:
            assert fold.n_val_clusters == 1
            assert fold.n_train_clusters == 4

    def test_two_clusters_two_folds(self):
        """Test with exactly 2 clusters and 2 folds."""
        rows = []
        for cluster_id in range(2):
            for _ in range(5):
                rows.append(
                    {
                        "cluster": cluster_id,
                        "TaskA": np.random.randn(),
                        "Quality": ["high", "low"][cluster_id],
                    }
                )
        df = pd.DataFrame(rows)

        folds = cluster_multilabel_stratified_kfold(
            df,
            cluster_col="cluster",
            task_cols=["TaskA"],
            quality_col="Quality",
            n_folds=2,
            shuffle=True,
            random_state=42,
        )

        assert len(folds) == 2

        # Each fold should have 1 train cluster and 1 val cluster
        for fold in folds:
            assert fold.n_train_clusters == 1
            assert fold.n_val_clusters == 1
            assert len(set(fold.train_clusters) & set(fold.val_clusters)) == 0


# ---------------------------------------------------------------------------
# Test: Validation set rotation across folds
# ---------------------------------------------------------------------------


class TestFoldRotation:
    """Verify that validation sets rotate correctly across folds."""

    def test_each_cluster_appears_in_validation_exactly_once(self, synthetic_clustered_df: pd.DataFrame):
        """In k-fold CV, each cluster should be in validation exactly once across all folds."""
        task_cols = ["TaskA", "TaskB", "TaskC"]

        folds = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        # Count how many times each cluster appears in validation
        val_counts: dict[int, int] = {}
        for fold in folds:
            for cluster_id in fold.val_clusters:
                val_counts[cluster_id] = val_counts.get(cluster_id, 0) + 1

        all_clusters = set(synthetic_clustered_df["cluster"].unique())

        # Each cluster should appear in validation exactly once
        for cluster_id in all_clusters:
            assert val_counts.get(cluster_id, 0) == 1, (
                f"Cluster {cluster_id} appears in validation {val_counts.get(cluster_id, 0)} times, "
                f"expected exactly 1 time across all folds."
            )

    def test_validation_sets_are_disjoint_across_folds(self, synthetic_clustered_df: pd.DataFrame):
        """Validation sets from different folds should be completely disjoint."""
        task_cols = ["TaskA", "TaskB", "TaskC"]

        folds = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        # Check all pairs of folds for validation set overlap
        for i, fold_i in enumerate(folds):
            for j, fold_j in enumerate(folds):
                if i >= j:
                    continue

                val_i = set(fold_i.val_indices)
                val_j = set(fold_j.val_indices)
                overlap = val_i & val_j

                assert len(overlap) == 0, (
                    f"Fold {i} and Fold {j} have {len(overlap)} overlapping validation indices. "
                    f"Validation sets must be disjoint across folds."
                )

    def test_union_of_validation_sets_equals_full_dataset(self, synthetic_clustered_df: pd.DataFrame):
        """Union of all validation sets should equal the entire dataset."""
        task_cols = ["TaskA", "TaskB", "TaskC"]
        all_indices = set(synthetic_clustered_df.index.tolist())

        folds = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        union_val = set()
        for fold in folds:
            union_val.update(fold.val_indices)

        assert union_val == all_indices, (
            f"Union of validation sets has {len(union_val)} indices, "
            f"expected {len(all_indices)}. Missing: {len(all_indices - union_val)}"
        )


# ---------------------------------------------------------------------------
# Test: Large-scale statistical validation
# ---------------------------------------------------------------------------


class TestLargeScaleValidation:
    """Statistical tests on larger datasets."""

    def test_repeated_splits_have_sufficient_variation(self, large_clustered_df: pd.DataFrame):
        """Multiple splits should show meaningful variation in fold composition."""
        task_cols = ["TaskA", "TaskB"]
        n_splits = 5

        all_fold_0_val_clusters: list[set] = []

        for split_idx in range(n_splits):
            folds = cluster_multilabel_stratified_kfold(
                large_clustered_df,
                cluster_col="cluster",
                task_cols=task_cols,
                quality_col="Quality",
                n_folds=5,
                shuffle=True,
                random_state=42 + split_idx,  # Different seed per split
            )
            all_fold_0_val_clusters.append(set(folds[0].val_clusters))

        # Count unique validation sets for fold 0 across splits
        unique_val_sets = len({frozenset(s) for s in all_fold_0_val_clusters})

        # With 50 clusters and different seeds, we expect variation
        assert unique_val_sets >= 3, (
            f"Only {unique_val_sets} unique validation sets for fold 0 across {n_splits} splits. "
            f"Expected more variation with different random seeds."
        )

    def test_cluster_integrity_maintained_at_scale(self, large_clustered_df: pd.DataFrame):
        """Verify cluster integrity on larger dataset with multiple splits."""
        task_cols = ["TaskA", "TaskB"]

        for split_idx in range(3):
            folds = cluster_multilabel_stratified_kfold(
                large_clustered_df,
                cluster_col="cluster",
                task_cols=task_cols,
                quality_col="Quality",
                n_folds=5,
                shuffle=True,
                random_state=42 + split_idx,
            )

            for fold in folds:
                # Check no cluster overlap
                train_clusters = set(fold.train_clusters)
                val_clusters = set(fold.val_clusters)
                assert len(train_clusters & val_clusters) == 0

                # Check all molecules in cluster have same assignment
                for cluster_id, group in large_clustered_df.groupby("cluster"):
                    mol_indices = set(group.index.tolist())
                    in_train = mol_indices & set(fold.train_indices)
                    in_val = mol_indices & set(fold.val_indices)
                    assert len(in_train) == 0 or len(in_val) == 0, (
                        f"Split {split_idx}, Fold {fold.fold_id}, Cluster {cluster_id}: "
                        f"Data leakage detected - cluster split between train and val"
                    )


# ---------------------------------------------------------------------------
# Test: Consistency of fold statistics
# ---------------------------------------------------------------------------


class TestFoldStatistics:
    """Verify fold statistics are correctly computed."""

    def test_molecule_counts_match_indices(self, synthetic_clustered_df: pd.DataFrame):
        """n_train_mols and n_val_mols should match actual index counts."""
        task_cols = ["TaskA", "TaskB", "TaskC"]

        folds = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        for fold in folds:
            assert fold.n_train_mols == len(fold.train_indices), (
                f"Fold {fold.fold_id}: n_train_mols ({fold.n_train_mols}) != "
                f"len(train_indices) ({len(fold.train_indices)})"
            )
            assert fold.n_val_mols == len(fold.val_indices), (
                f"Fold {fold.fold_id}: n_val_mols ({fold.n_val_mols}) != " f"len(val_indices) ({len(fold.val_indices)})"
            )

    def test_cluster_counts_match_arrays(self, synthetic_clustered_df: pd.DataFrame):
        """n_train_clusters and n_val_clusters should match actual cluster array lengths."""
        task_cols = ["TaskA", "TaskB", "TaskC"]

        folds = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        for fold in folds:
            assert fold.n_train_clusters == len(fold.train_clusters), (
                f"Fold {fold.fold_id}: n_train_clusters ({fold.n_train_clusters}) != "
                f"len(train_clusters) ({len(fold.train_clusters)})"
            )
            assert fold.n_val_clusters == len(fold.val_clusters), (
                f"Fold {fold.fold_id}: n_val_clusters ({fold.n_val_clusters}) != "
                f"len(val_clusters) ({len(fold.val_clusters)})"
            )

    def test_total_molecules_preserved(self, synthetic_clustered_df: pd.DataFrame):
        """Total molecules in train + val should equal dataset size."""
        task_cols = ["TaskA", "TaskB", "TaskC"]
        total_molecules = len(synthetic_clustered_df)

        folds = cluster_multilabel_stratified_kfold(
            synthetic_clustered_df,
            cluster_col="cluster",
            task_cols=task_cols,
            quality_col="Quality",
            n_folds=5,
            shuffle=True,
            random_state=42,
        )

        for fold in folds:
            fold_total = fold.n_train_mols + fold.n_val_mols
            assert fold_total == total_molecules, (
                f"Fold {fold.fold_id}: train ({fold.n_train_mols}) + val ({fold.n_val_mols}) = "
                f"{fold_total}, expected {total_molecules}"
            )
