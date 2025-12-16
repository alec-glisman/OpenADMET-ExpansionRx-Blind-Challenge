# %% [markdown]
# ## Setup

import gc
import itertools
import logging
import warnings
from datetime import datetime

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
import useful_rdkit_utils as uru
from datasets import Dataset, DatasetDict
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm.auto import tqdm

# %%
plt.style.use(["science"])

# %%
# setup tqdm
tqdm.pandas()

# %%
# setup logging
level = logging.DEBUG
logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(level)

logger.info("Imports successful.")

# %% [markdown]
# ## Load Data

# %%
# Data input and output directories
base_data_dir = Path().cwd().parents[0] / "assets/dataset/eda/data/set"
output_dir = base_data_dir.parents[2] / "splits"
output_dir.mkdir(parents=True, exist_ok=True)

output_fig_dir = output_dir / f"figures/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
output_fig_dir.mkdir(parents=True, exist_ok=True)

if not base_data_dir.exists():
    raise FileNotFoundError(f"Data directory not found at {base_data_dir}")

logger.info(f"Output directory set to {output_dir}")
logger.info(f"Input data directory found at {base_data_dir}")
for dataset_dir in base_data_dir.iterdir():
    logger.info(f"Dataset name: {dataset_dir.name}")

# %%
# Load input datasets
datasets = {
    "high": pd.read_csv(base_data_dir / "cleaned_combined_datasets_high_quality.csv"),
    "medium": pd.read_csv(base_data_dir / "cleaned_combined_datasets_medium_high_quality.csv", low_memory=False),
    "low": pd.read_csv(base_data_dir / "cleaned_combined_datasets_low_medium_high_quality.csv", low_memory=False),
}

for name, df in datasets.items():
    logger.info(f"Dataset: {name}, shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Unique Dataset Constituents: {df['Dataset'].unique()}")

# %%
# calculate fingerprints for all molecules in each dataset
fpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=3,
    countSimulation=False,
    includeChirality=False,
    fpSize=2048,
)

for name, df in datasets.items():
    logger.info(f"Calculating fingerprints for dataset: {name}")
    df["mol"] = df["SMILES"].progress_apply(Chem.MolFromSmiles)
    df["Fingerprint"] = df["mol"].progress_apply(fpgen.GetCountFingerprintAsNumPy)

    df.drop(columns=["mol"], inplace=True)

    # put fingerprint column after "Molecule Name,SMILES,Dataset"
    cols = df.columns.tolist()
    cols.insert(3, cols.pop(cols.index("Fingerprint")))
    df = df[cols]

    # expand fingerprint numpy arrays into separate columns
    fp_array = np.vstack(df["Fingerprint"].values)
    fp_df = pd.DataFrame(fp_array, columns=[f"Morgan_FP_{i}" for i in range(fp_array.shape[1])])
    df = pd.concat([df.reset_index(drop=True), fp_df.reset_index(drop=True)], axis=1)
    df.drop(columns=["Fingerprint"], inplace=True)
    logger.debug(f"Number of fingerprint columns added: {fp_df.shape[1]}")

    datasets[name] = df
    logger.info(f"Fingerprints calculated for dataset: {name}")
    logger.debug(f"Dataset {name} columns after fingerprint calculation: {df.columns.tolist()}")

# %%
# on high-quality dataset, sort by Molecule Name ascending and split test/train by first 90%/10%
percentage_train = 0.9
percentage_validation = 0.1

high_quality_df = datasets["high"].sort_values(by="Molecule Name").reset_index(drop=True)
n_total = high_quality_df.shape[0]
n_train = int(n_total * percentage_train)
n_test = n_total - n_train

test_df = high_quality_df.iloc[n_train:]
train_df = high_quality_df.iloc[:n_train]

# randomly split train into train/validation sets
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
n_validation = int(train_df.shape[0] * percentage_validation)
validation_df = train_df.iloc[:n_validation]
train_df = train_df.iloc[n_validation:]


logger.info(f"High-quality dataset total samples: {n_total}")
logger.info(f"Training samples: {train_df.shape[0]}")
logger.info(f"Validation samples: {validation_df.shape[0]}")
logger.info(f"Testing samples: {test_df.shape[0]}")

# save to temporal datasplit
temporal_dir = output_dir / "high_quality/temporal_split"
temporal_dir.mkdir(parents=True, exist_ok=True)

# convert to hf dataset and save
train_hf = Dataset.from_pandas(train_df, preserve_index=False)
validation_hf = Dataset.from_pandas(validation_df, preserve_index=False)
test_hf = Dataset.from_pandas(test_df, preserve_index=False)
temporal_hf = DatasetDict({"train": train_hf, "validation": validation_hf, "test": test_hf})
# save to disk
temporal_hf.save_to_disk(str(temporal_dir))
logger.info(f"Temporal split datasets saved to {temporal_dir}")

# %%
n_folds = 5
n_splits = 5
percentage_validation = 0.1
stratify_column = "Dataset"

split_dict = {
    "random_cluster": uru.get_random_clusters,
    "scaffold_cluster": uru.get_bemis_murcko_clusters,
    "kmeans_cluster": uru.get_kmeans_clusters,  # n_clusters = 10 by default
    "umap_cluster": uru.get_umap_clusters,  # n_clusters = 7 by default
    "butina_cluster": uru.get_butina_clusters,  # cutoff = 0.65 by default
}

# %%
split_datasets: dict[str, dict] = {}

n_iter = len(datasets) * len(split_dict) * n_splits
logger.info(f"Total iterations for dataset splits: {n_iter}")

pbar = tqdm(total=n_iter, desc="Creating dataset splits")
for dset_name, data in datasets.items():  # iterate over different datasets
    split_datasets[dset_name] = {}

    for split_name, split in split_dict.items():  # iterate over different splitting methods
        logger.info(f"Processing dataset: {dset_name} with split method: {split_name}")
        split_datasets[dset_name][split_name] = {}

        for i in range(0, n_splits):  # iterate over different splits
            split_datasets[dset_name][split_name][f"split_{i}"] = {}
            group_kfold_shuffle = uru.GroupKFoldShuffle(n_splits=n_folds, random_state=i, shuffle=True)

            for group in data[stratify_column].unique():  # iterate over different dataset groups
                # stratified group k-fold split (based on "Dataset" column)
                subdata = data[data[stratify_column] == group]
                cluster_list = split(subdata.SMILES)

                # make fictitious subdata indices to map back to original data later
                subdata_indices = subdata.index.to_numpy().copy()

                # iterate over different folds within each split
                for j, (subdata_train_idx, subdata_test_idx) in tqdm(
                    enumerate(
                        group_kfold_shuffle.split(subdata_indices, groups=cluster_list),
                    ),
                    desc=f"Dataset: {dset_name}, Split: {split_name}, Group: {group}",
                    leave=False,
                ):

                    if f"fold_{j}" not in split_datasets[dset_name][split_name][f"split_{i}"]:
                        split_datasets[dset_name][split_name][f"split_{i}"][f"fold_{j}"] = {}

                    # map indices back to original data
                    train_idx = subdata_indices[subdata_train_idx]
                    test_idx = subdata_indices[subdata_test_idx]

                    # further split train_idx into train and validation sets
                    n_train_samples = len(train_idx)
                    n_val_samples = int(n_train_samples * percentage_validation)
                    np.random.seed(i + j)  # ensure reproducibility
                    shuffled_train_idx = np.random.permutation(train_idx)
                    val_idx = shuffled_train_idx[:n_val_samples]
                    train_idx = shuffled_train_idx[n_val_samples:]

                    # save indices for each group split
                    split_datasets[dset_name][split_name][f"split_{i}"][f"fold_{j}"][group] = {
                        "train": train_idx,
                        "validation": val_idx,
                        "test": test_idx,
                    }

                    # garbage collection
                    gc.collect()

                # garbage collection
                gc.collect()

            pbar.update(1)

            # combine group splits into final train/test sets for each fold
            # logger.debug(f"Combining group splits for dataset: {dset_name}, split: {split_name}, iteration: {i}")
            for j in range(n_folds):
                if f"fold_{j}" not in split_datasets[dset_name][split_name][f"split_{i}"]:
                    raise ValueError(
                        f"Fold {j} not found in split {i} for dataset {dset_name} and split method {split_name}"
                    )

                combined_train_indices: list = []
                combined_val_indices: list = []
                combined_test_indices: list = []

                for group in data[stratify_column].unique():
                    group_split = split_datasets[dset_name][split_name][f"split_{i}"][f"fold_{j}"][group]
                    combined_train_indices.extend(group_split["train"])
                    combined_val_indices.extend(group_split["validation"])
                    combined_test_indices.extend(group_split["test"])

                combined_train_arr = np.array(combined_train_indices)
                combined_val_arr = np.array(combined_val_indices)
                combined_test_arr = np.array(combined_test_indices)

                # save combined train/test sets
                split_datasets[dset_name][split_name][f"split_{i}"][f"fold_{j}"]["total"] = {
                    "train": combined_train_arr,
                    "validation": combined_val_arr,
                    "test": combined_test_arr,
                }

                # final assertions


pbar.close()

# %%
# save all datasets with name format {dataset}_quality/{split_method}/split{split_number}_fold{fold_number}.csv
for dset_name, splits in split_datasets.items():
    for split_name, split_data in splits.items():
        for split_number, folds in split_data.items():
            for fold_number, datasets_dict in folds.items():

                split_output_dir = output_dir / f"{dset_name}_quality/{split_name}/{split_number}/{fold_number}"
                split_output_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Saving dataset to {split_output_dir}")

                train_idx = datasets_dict["total"]["train"]
                val_idx = datasets_dict["total"]["validation"]
                test_idx = datasets_dict["total"]["test"]

                # convert pandas to HF dataset
                train_hf = Dataset.from_pandas(data.loc[train_idx], preserve_index=False)
                val_hf = Dataset.from_pandas(data.loc[val_idx], preserve_index=False)
                test_hf = Dataset.from_pandas(data.loc[test_idx], preserve_index=False)
                dset = DatasetDict({"train": train_hf, "validation": val_hf, "test": test_hf})

                # Save to disk as HF dataset
                dset.save_to_disk(f"{split_output_dir}/hf_dataset")

        # print size of folder in MB after saving all splits
        folder_size = sum(f.stat().st_size for f in split_output_dir.glob("**/*") if f.is_file())
        folder_size_mb = folder_size / (1024 * 1024)
        logger.info(
            f"Saved all splits for dataset: {dset_name}, split method: {split_name}. Folder size: {folder_size_mb:.2f} MB"
        )

# %%
# plot the number of samples for each endpoint in each split
for dset_name, splits in split_datasets.items():
    for split_name, split_data in splits.items():
        for split_number, folds in split_data.items():
            for fold_number, datasets_dict in folds.items():
                split_output_dir = output_dir / f"{dset_name}_quality/{split_name}/{split_number}/{fold_number}"

                train_idx = datasets_dict["total"]["train"]
                val_idx = datasets_dict["total"]["validation"]
                test_idx = datasets_dict["total"]["test"]

                train_df = datasets[dset_name].loc[train_idx]
                val_df = datasets[dset_name].loc[val_idx]
                test_df = datasets[dset_name].loc[test_idx]

                # count number of samples for each endpoint
                endpoints = [
                    "LogD",
                    "KSOL",
                    "HLM CLint",
                    "MLM CLint",
                    "Caco-2 Permeability Papp A>B",
                    "Caco-2 Permeability Efflux",
                    "MPPB",
                    "MBPB",
                    "MGMB",
                ]

                counts = {
                    "train": [train_df[ep].notnull().sum() for ep in endpoints],
                    "validation": [val_df[ep].notnull().sum() for ep in endpoints],
                    "test": [test_df[ep].notnull().sum() for ep in endpoints],
                }

                counts_df = pd.DataFrame(counts, index=endpoints)

                # plot
                ax = counts_df.plot.bar(rot=45, figsize=(10, 6))
                ax.set_title(
                    f"Dataset: {dset_name}, Split: {split_name}, {split_number}, {fold_number} - Sample Counts per Endpoint"
                )
                ax.set_ylabel("Number of Samples")
                plt.tight_layout()
                plt_path = split_output_dir / "sample_counts_per_endpoint.png"
                plt.savefig(plt_path, dpi=600)
                plt.close()
                logger.info(f"Saved sample counts plot to {plt_path}")

# %%
# boxplot for number of test samples for each split method (x) on different datasets (separate plots)
for dset_name, splits in split_datasets.items():
    logger.info(f"Creating boxplot for dataset: {dset_name}")

    plot_data = []
    for split_name, split_data in splits.items():
        for split_id, folds in split_data.items():
            for fold_id, datasets in folds.items():
                n_test_samples = len(datasets["total"]["test"])
                plot_data.append({"Split Method": split_name, "Number of Test Samples": n_test_samples})
    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Split Method", y="Number of Test Samples", data=plot_df, ax=ax)
    ax.set_title(f"Distribution of Test Set Sizes for Different Split Methods\nDataset: {dset_name}")
    ax.set_ylabel("Number of Test Samples")
    ax.set_xlabel("Split Method")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_fig_dir / f"{dset_name}_test_set_size_distribution.png", dpi=600)

# %%
# boxplot distribution of data points for each split method and each dataset
for dset_name, splits in split_datasets.items():
    for split_name, split_data in splits.items():
        fold_sizes = []
        for split_id, folds in split_data.items():
            for fold_id, groups in folds.items():
                for group_name, datasets in groups.items():
                    # ignore "total"
                    if group_name == "total":
                        continue

                    train_size = len(datasets["train"])
                    test_size = len(datasets["test"])
                    fold_sizes.append(
                        {
                            "Split ID": split_id,
                            "Fold ID": fold_id,
                            "Group": group_name,
                            "Train Size": train_size,
                            "Test Size": test_size,
                        }
                    )
        fold_sizes_df = pd.DataFrame(fold_sizes)

        # 1 figure with 2 boxplots: train size and test size
        logger.info(
            f"Creating train/test size distribution boxplots for dataset: {dset_name}, split method: {split_name}"
        )
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        sns.boxplot(x="Group", y="Train Size", data=fold_sizes_df, ax=axs[0])
        axs[0].set_title(
            f"Train Set Size Distribution: {dset_name.capitalize()} Quality, {split_name.replace('_', ' ').capitalize()} Split"
        )

        sns.boxplot(x="Group", y="Test Size", data=fold_sizes_df, ax=axs[1])
        axs[1].set_title(
            f"Test Set Size Distribution: {dset_name.capitalize()} Quality, {split_name.replace('_', ' ').capitalize()} Split"
        )

        for ax in axs:
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylabel("Number of Data Points")
            ax.set_xlabel("Provenance")

        fig.tight_layout()
        fig.savefig(
            output_fig_dir / f"{dset_name}_quality_{split_name}_split_train_test_size_distribution_boxplot.png",
            dpi=600,
        )
