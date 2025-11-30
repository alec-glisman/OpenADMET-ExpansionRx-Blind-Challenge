task_cols = [f"target_{i}" for i in range(9)]

# Stratified, cluster-aware K-fold with diagnostics + plots
folds, diag = cluster_multilabel_stratified_kfold(
    df,
    cluster_col="bitbirch_cluster",
    task_cols=task_cols,
    n_splits=5,
    diagnostics=True,
    make_plots=True,
)

for f in folds:
    print(
        f"Fold {f.fold_id}: "
        f"{f.n_train_mols} train mols ({f.n_train_clusters} clusters), "
        f"{f.n_val_mols} val mols ({f.n_val_clusters} clusters)"
    )

# If running in a notebook / script, you can show the figures:
# diag.hist_fig.show()
# diag.boxplot_fig.show()
