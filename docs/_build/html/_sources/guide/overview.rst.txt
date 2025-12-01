Overview
========

Welcome to the OpenADMET + ExpansionRx Blind Challenge! This project provides a comprehensive toolkit for building, training, and evaluating predictive models for ADMET endpoints. Whether you are a participant in the challenge or a researcher looking for a robust ADMET modeling framework, this documentation will guide you through the process.

The challenge focuses on leveraging open-source tooling, multitask learning, and transfer learning to improve predictive performance on curated ADMET datasets.

Key Components
--------------

The framework is organized into several modular components, each designed to handle a specific part of the machine learning lifecycle:

* **Data Pipeline** (``admet.data``): Handles data ingestion, harmonization, and standardization. Includes tools for molecular fingerprint generation, cluster-based dataset splitting with BitBirch, and quality-aware stratification.
* **Chemprop Models** (``admet.model.chemprop``): Message-passing neural networks using Chemprop with PyTorch Lightning. Supports ensemble training, hyperparameter optimization with Ray Tune, and curriculum learning.
* **Classical Models** (``admet.model.classical``): Traditional ML models (XGBoost, LightGBM) for baseline comparisons.
* **Visualization** (``admet.plot``): Parity plots, metric bar charts, distribution plots, and publication-ready figure generation.

Getting Started
---------------

If you are new to the project, we recommend starting with the :doc:`modeling` guide to learn how to train Chemprop models. For those interested in contributing or extending the framework, the :doc:`development` guide provides essential setup instructions.
