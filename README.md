# OpenADMET + ExpansionRx Blind Challenge

**Authors**: Alec Glisman, PhD
**Date**: October 2025

## Goals

#### Models

We plan to benchmark the following models:

- XGBoost Baseline
- Chemprop Multitask (trained from scratch)
- Chemprop CheMeleon (finetuned)
- GROVER (finetuned)
- KERMT (finetuned)
- ChemBERTa-3 (finetuned)

These models were selected to represent a range of architectures and training paradigms, including tree-based methods, graph neural networks, and transformer-based models. We will explore both training from scratch and transfer learning approaches.

Each of the models will be trained as ensemble models with 5-fold cross-validation to improve robustness and performance, as recommended in the literature (Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery). Each fold should have multiple random seeds to further enhance ensemble diversity. By default, we will try five random seeds per fold, resulting in a total of 25 models per endpoint per architecture.

We will explore various hyperparameter optimization strategies, including grid search, random search, and Bayesian optimization, to identify the best model configurations for each architecture.

We will also explore super-ensembling techniques to combine the embeddings and/or predictions from multiple models to further enhance predictive performance.

### Endpoints

We are predicting the following ADMET endpoints:

| Column                       | Unit        | Type      | Description                                   |
|:---------------------------- |:----------: |:--------: |:----------------------------------------------|
| Molecule Name                |             |    str    | Identifier for the molecule |
| Smiles                       |             |    str    | Text representation of the 2D molecular structure |
| LogD                         |             |   float   | LogD calculation |
| KSol                         |    uM       |   float   | Kinetic Solubility |
| MLM CLint                    | mL/min/kg   |   float   | Mouse Liver Microsomal |
| HLM CLint                    | mL/min/kg   |   float   | Human Liver Microsomal |
| Caco-2 Permeability Efflux   |             |   float   | Caco-2 Permeability Efflux |
| Caco-2 Permeability Papp A>B | 10^-6 cm/s  |   float   | Caco-2 Permeability Papp A>B |
| MPPB                         | % Unbound   |   float   | Mouse Plasma Protein Binding |
| MBPB                         | % Unbound   |   float   | Mouse Brain Protein Binding |
| MGMB.                        | % Unbound   |   float   | Mouse Gastrocnemius Muscle Binding |

The challenge will be judged based on the following criteria:

- We welcome submissions of any kind, including machine learning and physics-based approaches. You can also employ pre-training approaches as you see fit, as well as incorporate data from external sources into your models and submissions.
- In the spirit of open science and open source we would love to see code showing how you created your submission if possible, in the form of a Github Repository. If not possible due to IP or other constraints you must at a minimum provide a short report written methodology based on the template here. Make sure your lat submission before the deadline includes a link to a report or to a Github repository.
- Each participant can submit as many times as they like, up to a limit of once per day. Only your latest submission will be considered for the final leaderboard.
- The endpoints will be judged individually by mean absolute error (MAE), while an overall leaderboard will be judged by the macro-averaged relative absolute error (MA-RAE).
- For endpoints that are not already on a log scale (e.g LogD) they will be transformed to log scale to minimize the impact of outliers on evaluation.
- We will estimate errors on the metrics using bootstrapping and use the statistical testing workflow outlined in this paper to determine if model performance is statistically distinct.

### Datasets

We will attempt to augment the provided training dataset with additional publicly available ADMET datasets to improve model performance. Potential sources for augmentation are listed in the Links section below.

## Open Questions

- How to best split the dataset for train/validation/test?
  - Random split
  - Scaffold split
  - Butina clustering split
  - UMAP clustering split
  - Time-based split (if timestamps are available)
- Should we incorprate stereochemistry information or remove it as part of preprocessing?
- How should we handle salts and counterions in the SMILES strings?
- How should we handle tautomeric forms of molecules?

## Links

### Challenge Information

- [Challenge Hugging Face Page](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)
- [Teaser Dataset on Hugging Face](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-teaser)

    ```python
    # Hugging Face Datasets library
    from datasets import load_dataset
    ds = load_dataset("openadmet/openadmet-expansionrx-challenge-teaser")
    ```

- [Full Dataset on Hugging Face (not live)](https://huggingface.co/datasets/openadmet/openadmet-challenge-train-data)

### Models

- [XGBoost Baseline](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)
- [Chemprop Multitask](https://chemprop.readthedocs.io/en/latest/multi_task.html)
- [Chemprop Pretrained](https://chemprop.readthedocs.io/en/latest/chemeleon_foundation_finetuning.html)
- [ChemBERTa Foundation](https://deepchem.io/tutorials/transfer-learning-with-chemberta-transformers/)
- [KERMT Pretrained](https://github.com/NVIDIA-Digital-Bio/KERMT)

### External Datasets

- [KERMT](https://figshare.com/articles/dataset/Datasets_for_Multitask_finetuning_and_acceleration_of_chemical_pretrained_models_for_small_molecule_drug_property_prediction_/30350548/2)
- [Polaris Antiviral](https://polarishub.io/datasets/asap-discovery/antiviral-admet-2025-unblinded)
- [Polaris ADME Fang](https://polarishub.io/datasets/biogen/adme-fang-v1)
- [TDC](https://tdcommons.ai/benchmark/admet_group/overview/)
- [PharmaBench](https://github.com/mindrank-ai/PharmaBench)
- [NCATS](https://opendata.ncats.nih.gov/adme/data)
- [admetSAR 3.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC11223829/#:~:text=Data%20collection,are%20available%20in%20Text%20S2.)
- [admetics](https://github.com/datagrok-ai/admetica)

### Papers and Blogs

- [Dataset Splitting](https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html)
- [Benchmarking](https://practicalcheminformatics.blogspot.com/2023/08/we-need-better-benchmarks-for-machine.html)
- [Comparisons](https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html)
- [Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c01609)

### Coding Assistants

- [Copilot Prompts](https://github.com/github/awesome-copilot/tree/main?tab=readme-ov-file)
