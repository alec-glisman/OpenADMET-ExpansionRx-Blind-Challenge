from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuration
OUTPUT_DIR = Path("assets/eda/hpo_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_performance_distribution(df):
    """Plot the distribution of validation MAE."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df["val_mae"], kde=True, bins=30)
    plt.title("Distribution of Validation MAE")
    plt.xlabel("Validation MAE")
    plt.ylabel("Count")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.savefig(OUTPUT_DIR / "val_mae_distribution.png")
    plt.close()


def plot_hyperparameter_correlations(df, hparams):
    """Plot correlations between hyperparameters and validation MAE."""
    # Select numerical hyperparameters
    numeric_hparams = [h for h in hparams if pd.api.types.is_numeric_dtype(df[h])]

    # Add derived LRs to correlation analysis
    numeric_hparams.extend(["init_lr", "max_lr", "final_lr"])

    # Calculate correlations with val_mae
    correlations = df[numeric_hparams + ["val_mae"]].corr()["val_mae"].sort_values()

    plt.figure(figsize=(12, 8))
    correlations.drop("val_mae").plot(kind="barh")
    plt.title("Correlation of Hyperparameters with Validation MAE")
    plt.xlabel("Correlation Coefficient")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hyperparameter_correlations.png")
    plt.close()

    # Scatter plots for specific requested features + top correlated
    specific_features = ["config/dropout", "init_lr", "max_lr", "final_lr"]
    top_features = correlations.abs().sort_values(ascending=False).index.tolist()

    # Combine specific and top features, removing duplicates and val_mae
    plot_features = []
    seen = set()

    # Add specific features first
    for f in specific_features:
        if f in df.columns and f not in seen:
            plot_features.append(f)
            seen.add(f)

    # Add top correlated features until we have enough
    for f in top_features:
        if f != "val_mae" and f not in seen:
            plot_features.append(f)
            seen.add(f)

    # Take top 9 features for a 3x3 grid
    plot_features = plot_features[:9]

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()

    for i, feature in enumerate(plot_features):
        if i >= len(axes):
            break
        sns.scatterplot(data=df, x=feature, y="val_mae", ax=axes[i], alpha=0.6)
        axes[i].set_title(f"{feature} vs Val MAE")
        # Log scale for LRs if needed
        if "lr" in feature:
            axes[i].set_xscale("log")
        axes[i].grid(True, which="both", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_scatter_plots.png")
    plt.close()


def plot_categorical_performance(df, hparams):
    """Violin plots for categorical and discrete hyperparameters."""
    # Include categorical and low-cardinality numerical columns (e.g. < 20 unique values)
    # This will catch things like depth, num_layers, batch_size, etc.
    categorical_hparams = [h for h in hparams if not pd.api.types.is_numeric_dtype(df[h]) or df[h].nunique() < 20]

    for cat in categorical_hparams:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x=cat, y="val_mae")
        plt.title(f"Performance by {cat}")
        plt.xticks(rotation=45)

        # Add grid lines
        plt.grid(True, which="major", axis="y", linestyle="-", alpha=0.7)
        plt.minorticks_on()
        plt.grid(True, which="minor", axis="y", linestyle=":", alpha=0.4)

        plt.tight_layout()
        safe_name = cat.replace("/", "_")
        plt.savefig(OUTPUT_DIR / f"violinplot_{safe_name}.png")
        plt.close()


def plot_target_weights(df):
    """Plot validation MAE vs target weights."""
    target_weight_cols = [c for c in df.columns if "target_weight" in c]

    if not target_weight_cols:
        return

    n_cols = 3
    n_rows = (len(target_weight_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(target_weight_cols):
        if i >= len(axes):
            break
        sns.scatterplot(data=df, x=col, y="val_mae", ax=axes[i], alpha=0.6)
        axes[i].set_title(f"{col.replace('config/target_weight_', '')} vs Val MAE")
        axes[i].grid(True, which="both", linestyle="--", alpha=0.7)

    # Hide empty subplots
    for i in range(len(target_weight_cols), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_weights_scatter.png")
    plt.close()


def plot_clusters(X_scaled, clusters, selected_mask=None):
    """Visualize clusters using PCA."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab20", alpha=0.6, label="Candidates")

    if selected_mask is not None:
        plt.scatter(
            X_pca[selected_mask, 0],
            X_pca[selected_mask, 1],
            c="red",
            marker="*",
            s=200,
            label="Selected",
            edgecolors="black",
        )

    plt.title("PCA Visualization of Hyperparameter Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.savefig(OUTPUT_DIR / "cluster_visualization.png")
    plt.close()


def plot_comprehensive_correlations(df, hparams):
    """Plot a comprehensive correlation heatmap of all numerical features."""
    # Select numerical hyperparameters
    numeric_hparams = [h for h in hparams if pd.api.types.is_numeric_dtype(df[h])]
    numeric_hparams.extend(["init_lr", "max_lr", "final_lr"])

    # Add target weights
    target_weight_cols = [c for c in df.columns if "target_weight" in c]

    # Combine all features
    all_features = numeric_hparams + target_weight_cols + ["val_mae"]

    # Calculate correlation matrix
    corr_matrix = df[all_features].corr()

    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.title("Comprehensive Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comprehensive_correlation_heatmap.png")
    plt.close()


def plot_all_distributions(df, hparams):
    """Plot distributions for ALL hyperparameters."""
    # Numerical distributions
    numeric_hparams = [h for h in hparams if pd.api.types.is_numeric_dtype(df[h])]
    numeric_hparams.extend(["init_lr", "max_lr", "final_lr"])

    n_cols = 4
    n_rows = (len(numeric_hparams) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_hparams):
        if i >= len(axes):
            break
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col.replace('config/', '')}")
        if "lr" in col:
            axes[i].set_xscale("log")
        axes[i].grid(True, which="both", linestyle="--", alpha=0.7)

    for i in range(len(numeric_hparams), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_numeric_distributions.png")
    plt.close()

    # Categorical distributions
    categorical_hparams = [h for h in hparams if not pd.api.types.is_numeric_dtype(df[h]) or df[h].nunique() < 20]

    n_rows = (len(categorical_hparams) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_hparams):
        if i >= len(axes):
            break
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f"Count of {col.replace('config/', '')}")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].grid(True, which="both", linestyle="--", alpha=0.7)

    for i in range(len(categorical_hparams), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_categorical_distributions.png")
    plt.close()


def plot_pairwise_interactions(df, hparams):
    """Plot pairwise interactions for top correlated features."""
    # Select top 5 numerical features correlated with val_mae
    numeric_hparams = [h for h in hparams if pd.api.types.is_numeric_dtype(df[h])]
    numeric_hparams.extend(["init_lr", "max_lr", "final_lr"])

    correlations = df[numeric_hparams + ["val_mae"]].corr()["val_mae"].abs().sort_values(ascending=False)
    top_features = correlations.index[1:6].tolist()  # Top 5 excluding val_mae

    plot_cols = top_features + ["val_mae"]

    sns.pairplot(df[plot_cols], diag_kind="kde", corner=True)
    plt.suptitle("Pairwise Interactions of Top Features", y=1.02)
    plt.savefig(OUTPUT_DIR / "pairwise_interactions.png")
    plt.close()


def plot_parallel_coordinates(df, hparams):
    """Plot parallel coordinates for top performing models."""
    from pandas.plotting import parallel_coordinates

    # Select top 20 models
    top_df = df.sort_values("val_mae").head(20).copy()

    # Select numerical features to visualize
    features = ["config/depth", "config/message_hidden_dim", "config/dropout", "config/ffn_num_layers", "val_mae"]

    # Normalize data for better visualization
    scaler = StandardScaler()
    top_df_norm = pd.DataFrame(scaler.fit_transform(top_df[features]), columns=features)

    # Add a class column for coloring (e.g., quartiles of MAE)
    top_df_norm["performance"] = pd.qcut(top_df["val_mae"], q=4, labels=["Best", "Good", "Average", "Poor"])

    plt.figure(figsize=(15, 8))
    parallel_coordinates(top_df_norm, "performance", colormap=plt.get_cmap("viridis"))
    plt.title("Parallel Coordinates of Top 20 Models (Normalized)")
    plt.ylabel("Normalized Value (Standard Deviations)")
    plt.xticks(rotation=45)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "parallel_coordinates.png")
    plt.close()


def plot_target_weight_correlations(df):
    """Plot correlations between target weights."""
    target_weight_cols = [c for c in df.columns if "target_weight" in c]

    if not target_weight_cols:
        return

    corr_matrix = df[target_weight_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Target Weight Correlations")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_weight_correlations.png")
    plt.close()


def plot_feature_importance(df, hparams):
    """Plot feature importance using Random Forest."""
    from sklearn.ensemble import RandomForestRegressor

    # Prepare data
    X = df[hparams].copy()
    y = df["val_mae"]

    # Handle categorical variables
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get importances
    importances = pd.Series(rf.feature_importances_, index=hparams).sort_values(ascending=True)

    plt.figure(figsize=(12, 8))
    importances.plot(kind="barh")
    plt.title("Hyperparameter Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_rf.png")
    plt.close()


def plot_best_vs_worst(df, hparams):
    """Compare hyperparameter distributions of best 10% vs worst 10% models."""
    n = int(len(df) * 0.1)
    best_df = df.head(n).copy()
    worst_df = df.tail(n).copy()

    best_df["group"] = "Best 10%"
    worst_df["group"] = "Worst 10%"
    combined = pd.concat([best_df, worst_df])

    # Select numerical params
    numeric_hparams = [h for h in hparams if pd.api.types.is_numeric_dtype(df[h])]
    numeric_hparams.extend(["init_lr", "max_lr", "final_lr"])

    n_cols = 4
    n_rows = (len(numeric_hparams) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_hparams):
        if i >= len(axes):
            break
        sns.violinplot(data=combined, x="group", y=col, ax=axes[i])
        axes[i].set_title(f"{col.replace('config/', '')}")
        if "lr" in col:
            axes[i].set_yscale("log")
        axes[i].grid(True, which="both", linestyle="--", alpha=0.7)

    for i in range(len(numeric_hparams), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_vs_worst_comparison.png")
    plt.close()


def plot_cluster_performance(df_top):
    """Plot performance distribution by cluster."""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_top, x="cluster", y="val_mae")
    plt.title("Validation MAE Distribution by Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Validation MAE")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_performance.png")
    plt.close()


def plot_3d_scatter(df, hparams):
    """Plot 3D scatter of top 3 important features vs performance."""
    # Identify top 3 numeric features
    numeric_hparams = [h for h in hparams if pd.api.types.is_numeric_dtype(df[h])]
    correlations = df[numeric_hparams + ["val_mae"]].corr()["val_mae"].abs().sort_values(ascending=False)
    top_3 = correlations.index[1:4].tolist()

    if len(top_3) < 3:
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(df[top_3[0]], df[top_3[1]], df[top_3[2]], c=df["val_mae"], cmap="viridis_r", s=50, alpha=0.8)

    ax.set_xlabel(top_3[0].replace("config/", ""))
    ax.set_ylabel(top_3[1].replace("config/", ""))
    ax.set_zlabel(top_3[2].replace("config/", ""))

    plt.colorbar(sc, label="Validation MAE")
    plt.title(f"3D Interaction: {top_3[0]} vs {top_3[1]} vs {top_3[2]}")
    plt.savefig(OUTPUT_DIR / "3d_scatter_top_features.png")
    plt.close()


def plot_interaction_heatmaps(df, hparams):
    """Plot 2D interaction heatmaps for top feature pairs."""
    numeric_hparams = [h for h in hparams if pd.api.types.is_numeric_dtype(df[h])]
    correlations = df[numeric_hparams + ["val_mae"]].corr()["val_mae"].abs().sort_values(ascending=False)
    top_features = correlations.index[1:5].tolist()

    import itertools

    pairs = list(itertools.combinations(top_features, 2))[:4]  # Top 4 pairs

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i, (f1, f2) in enumerate(pairs):
        if i >= len(axes):
            break

        # Bin data
        x_bins = pd.cut(df[f1], bins=10)
        y_bins = pd.cut(df[f2], bins=10)

        pivot = df.pivot_table(index=y_bins, columns=x_bins, values="val_mae", aggfunc="mean")

        sns.heatmap(pivot, ax=axes[i], cmap="viridis_r", annot=True, fmt=".3f")
        axes[i].set_title(f"Interaction: {f1} vs {f2}")
        axes[i].set_xlabel(f1)
        axes[i].set_ylabel(f2)
        axes[i].invert_yaxis()  # Match standard plot orientation

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interaction_heatmaps.png")
    plt.close()


def plot_tsne_manifold(X_scaled, df):
    """Visualize high-dimensional space using t-SNE."""
    from sklearn.manifold import TSNE

    # Use a subset if data is too large, but 500 is fine
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) - 1))
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df["val_mae"], cmap="viridis_r", alpha=0.7)
    plt.colorbar(scatter, label="Validation MAE")
    plt.title("t-SNE Manifold Visualization (colored by Performance)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.savefig(OUTPUT_DIR / "tsne_manifold.png")
    plt.close()


def plot_decision_tree_rules(df, hparams):
    """Visualize a decision tree to show explicit rules."""
    from sklearn.tree import DecisionTreeRegressor, plot_tree

    X = df[hparams].copy()
    y = df["val_mae"]

    # Handle categorical
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Train a shallow tree for interpretability
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(X, y)

    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=hparams, filled=True, rounded=True, precision=4, fontsize=10)
    plt.title("Decision Tree Rules for Validation MAE")
    plt.savefig(OUTPUT_DIR / "decision_tree_rules.png")
    plt.close()


def generate_ensemble_configs(selected_df, output_dir):
    """Generate YAML config files for the selected models."""
    configs_dir = output_dir / "ensemble_configs"
    configs_dir.mkdir(exist_ok=True)

    print(f"Generating {len(selected_df)} ensemble config files in {configs_dir}...")

    for i, (idx, row) in enumerate(selected_df.iterrows()):
        # Extract hyperparameters
        depth = int(row["config/depth"])
        message_hidden_dim = int(row["config/message_hidden_dim"])
        ffn_num_layers = int(row["config/ffn_num_layers"])
        ffn_hidden_dim = int(row["config/ffn_hidden_dim"])
        dropout = float(row["config/dropout"])
        batch_size = int(row["config/batch_size"])
        ffn_type = str(row["config/ffn_type"])
        aggregation = str(row["config/aggregation"])

        # Handle potential missing columns with defaults
        trunk_hidden_dim = int(row.get("config/hidden_dim", 500))

        # Learning rates
        max_lr = float(row["config/learning_rate"])
        warmup_ratio = float(row["config/lr_warmup_ratio"])
        final_ratio = float(row["config/lr_final_ratio"])

        init_lr = max_lr * warmup_ratio
        final_lr = max_lr * final_ratio

        # Target weights
        ordered_weights = []
        targets = [
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

        # Map display names to dataframe column suffixes
        target_mapping = {
            "LogD": "LogD",
            "Log KSOL": "Log_KSOL",
            "Log HLM CLint": "Log_HLM_CLint",
            "Log MLM CLint": "Log_MLM_CLint",
            "Log Caco-2 Permeability Papp A>B": "Log_Caco-2_Permeability_Papp_AgtB",
            "Log Caco-2 Permeability Efflux": "Log_Caco-2_Permeability_Efflux",
            "Log MPPB": "Log_MPPB",
            "Log MBPB": "Log_MBPB",
            "Log MGMB": "Log_MGMB",
        }

        for target in targets:
            # Construct expected column name
            suffix = target_mapping.get(target, target.replace(" ", "_"))
            col_name = f"config/target_weight_{suffix}"

            if col_name in row:
                ordered_weights.append(to_3sf(float(row[col_name])))
            else:
                # Fallback: try to find it if the name is slightly different
                found = False
                for c in row.index:
                    if "target_weight" in c and suffix in c:
                        ordered_weights.append(to_3sf(float(row[c])))
                        found = True
                        break
                if not found:
                    print(f"Warning: Could not find weight for {target} (expected {col_name})")
                    # Default to 1.0 if not found
                    ordered_weights.append(1.0)

        config = {
            "data": {
                "data_dir": "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data",
                "test_file": "assets/dataset/set/local_test.csv",
                "blind_file": "assets/dataset/set/blind_test.csv",
                "output_dir": None,
                "smiles_col": "SMILES",
                "target_cols": targets,
                "target_weights": ordered_weights,
                "splits": None,
                "folds": None,
            },
            "model": {
                "depth": depth,
                "message_hidden_dim": message_hidden_dim,
                "num_layers": ffn_num_layers,
                "hidden_dim": ffn_hidden_dim,
                "dropout": to_3sf(dropout),
                "batch_norm": True,
                "ffn_type": ffn_type,
                "aggregation": aggregation,
                "trunk_n_layers": 2,  # Default
                "trunk_hidden_dim": trunk_hidden_dim,
                "n_experts": 4,  # Default
            },
            "optimization": {
                "criterion": "MSE",
                "init_lr": ExponentialFloat(init_lr),
                "max_lr": ExponentialFloat(max_lr),
                "final_lr": ExponentialFloat(final_lr),
                "warmup_epochs": 5,
                "max_epochs": 150,
                "patience": 15,
                "batch_size": batch_size,
                "num_workers": 0,
                "seed": 12345,
                "progress_bar": False,
            },
            "mlflow": {
                "tracking": True,
                "tracking_uri": "http://127.0.0.1:8080",
                "experiment_name": "ensemble_chemprop",
                "run_name": None,
                "nested": True,
            },
        }

        # Write to file
        filename = configs_dir / f"ensemble_config_{i+1:02d}_rank_{i+1}.yaml"
        with open(filename, "w") as f:
            # Add header comments
            f.write("# Auto-generated ensemble config\n")
            f.write(f"# Rank: {i+1}\n")
            f.write(f"# Trial ID: {row['trial_id']}\n")
            f.write(f"# Reason: {row.get('reason', 'N/A')}\n")
            f.write(f"# Val MAE: {row['val_mae']:.4f}\n\n")

            try:
                yaml.dump(config, f, sort_keys=False, default_flow_style=False)
            except Exception as e:
                print(f"Error dumping YAML: {e}")
                # Fallback to string representation if yaml fails (unlikely)
                f.write(str(config))


class ExponentialFloat(float):
    pass


def exponential_float_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:float", f"{data:.2e}")


yaml.add_representer(ExponentialFloat, exponential_float_representer)


def to_3sf(x):
    if x == 0:
        return 0.0
    return float(f"{x:.3g}")


def main():
    # Load data
    csv_path = (
        "/media/aglisman/Linux_Overflow/home/aglisman/VSCodeProjects/"
        "OpenADMET-ExpansionRx-Blind-Challenge/assets/models/chemprop/hpo/chemprop_example/hpo_results.csv"
    )
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter valid runs
    df = df.dropna(subset=["val_mae"])
    df = df.sort_values("val_mae", ascending=True)
    print(f"Found {len(df)} valid runs.")

    # Hyperparameters to consider for diversity
    hparams = [
        "config/learning_rate",
        "config/lr_warmup_ratio",
        "config/lr_final_ratio",
        "config/dropout",
        "config/depth",
        "config/message_hidden_dim",
        "config/hidden_dim",
        "config/ffn_num_layers",
        "config/ffn_hidden_dim",
        "config/batch_size",
        "config/ffn_type",
        "config/aggregation",
    ]

    # Calculate derived LRs for plotting
    df["init_lr"] = df["config/learning_rate"] * df["config/lr_warmup_ratio"]
    df["max_lr"] = df["config/learning_rate"]
    df["final_lr"] = df["config/learning_rate"] * df["config/lr_final_ratio"]

    # Generate initial EDA plots
    print("Generating EDA plots...")
    plot_performance_distribution(df)
    plot_hyperparameter_correlations(df, hparams)
    plot_categorical_performance(df, hparams)
    plot_target_weights(df)

    # Generate comprehensive plots
    print("Generating comprehensive plots...")
    plot_comprehensive_correlations(df, hparams)
    plot_all_distributions(df, hparams)
    plot_pairwise_interactions(df, hparams)
    plot_parallel_coordinates(df, hparams)
    plot_target_weight_correlations(df)
    plot_feature_importance(df, hparams)
    plot_best_vs_worst(df, hparams)
    plot_3d_scatter(df, hparams)
    plot_interaction_heatmaps(df, hparams)
    plot_decision_tree_rules(df, hparams)

    # Preprocess for clustering
    X = df[hparams].copy()

    # Encode categorical variables
    le_ffn = LabelEncoder()
    X["config/ffn_type"] = le_ffn.fit_transform(X["config/ffn_type"].astype(str))

    le_agg = LabelEncoder()
    X["config/aggregation"] = le_agg.fit_transform(X["config/aggregation"].astype(str))

    # Normalize numerical variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Plot t-SNE
    plot_tsne_manifold(X_scaled, df)

    # Select top N models to cluster (e.g., top 100 or top 50% to ensure quality)
    top_n = min(100, len(df))
    df_top = df.head(top_n).copy()
    X_top = X_scaled[:top_n]

    # We want 20 configs total.
    # 1 is the absolute best.
    # 19 will be from clustering the top N models.

    # Cluster into 20 groups
    print(f"Clustering top {top_n} models into 20 groups...")
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_top)
    df_top["cluster"] = clusters

    # Plot cluster performance
    plot_cluster_performance(df_top)

    # Select best model from each cluster
    selected_indices = []
    selected_reasons = []

    # Always include the absolute best model
    best_idx = df.index[0]
    selected_indices.append(best_idx)
    selected_reasons.append("Best overall model (lowest val_mae)")

    # Get best from each cluster (excluding the one we already picked if it overlaps)
    for i in range(19):
        cluster_group = df_top[df_top["cluster"] == i]
        if cluster_group.empty:
            continue

        # Pick the one with lowest val_mae in the cluster
        best_in_cluster = cluster_group.sort_values("val_mae").iloc[0]

        if best_in_cluster.name != best_idx:
            selected_indices.append(best_in_cluster.name)
            # Generate a reason based on its distinctive features
            reason = f"Cluster {i+1} representative. "
            reason += f"Type: {best_in_cluster['config/ffn_type']}, "
            reason += f"Depth: {best_in_cluster['config/depth']}, "
            reason += f"Hidden: {best_in_cluster['config/message_hidden_dim']}"
            selected_reasons.append(reason)

    # If we have duplicates or fewer than 20, fill up with next best models
    unique_indices = list(dict.fromkeys(selected_indices))  # remove duplicates while preserving order

    if len(unique_indices) < 20:
        for idx in df.index:
            if idx not in unique_indices:
                unique_indices.append(idx)
                selected_reasons.append(f"Next best performing model (Rank {len(unique_indices)})")
                if len(unique_indices) == 20:
                    break

    # Visualize clusters and selection
    # Create a mask for selected items within the top N set
    # Note: selected_indices refers to the original df index
    # We need to map this back to the X_top array indices

    # Get indices of selected items that are in the top N
    top_n_indices = df_top.index
    selected_mask = [i for i, idx in enumerate(top_n_indices) if idx in unique_indices]

    print("Generating cluster visualization...")
    plot_clusters(X_top, clusters, selected_mask)

    # Output the selected configs
    selected_df = df.loc[unique_indices]
    selected_df["reason"] = selected_reasons

    print(f"Selected {len(selected_df)} configurations.")
    print(f"Plots saved to {OUTPUT_DIR.absolute()}")

    for i, (idx, row) in enumerate(selected_df.iterrows()):
        print(f"\n--- Config {i+1} ---")
        print(f"Reason: {row['reason']}")
        print(f"Trial ID: {row['trial_id']}")
        # idx is the index from the original dataframe, which might be an integer or not.
        # Assuming it's an integer index from read_csv
        print(f"Row Index: {idx}")
        print(f"val_mae: {row['val_mae']:.4f}")

        # Calculate derived LRs
        max_lr = row.get("config/learning_rate", 0.001)
        warmup_ratio = row.get("config/lr_warmup_ratio", 0.1)
        final_ratio = row.get("config/lr_final_ratio", 0.1)

        init_lr = max_lr * warmup_ratio
        final_lr = max_lr * final_ratio

        # Print relevant hyperparameters
        print("Hyperparameters:")
        print(f"  init_lr: {init_lr:.2e}")
        print(f"  max_lr: {max_lr:.2e}")
        print(f"  final_lr: {final_lr:.2e}")

        for hp in hparams:
            # Skip the raw LR params we just handled
            if hp in ["config/learning_rate", "config/lr_warmup_ratio", "config/lr_final_ratio"]:
                continue
            print(f"  {hp.replace('config/', '')}: {row[hp]}")

        # Print target weights
        print("Target Weights:")
        for col in row.index:
            if "target_weight" in col:
                print(f"  {col.replace('config/target_weight_', '')}: {row[col]:.4f}")

    # Generate YAML configs
    generate_ensemble_configs(selected_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
