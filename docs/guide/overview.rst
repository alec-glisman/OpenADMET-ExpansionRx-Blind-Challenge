Overview
========

Welcome to the OpenADMET + ExpansionRx Blind Challenge! This project provides a comprehensive toolkit for building, training, and evaluating predictive models for ADMET endpoints. Whether you are a participant in the challenge or a researcher looking for a robust ADMET modeling framework, this documentation will guide you through the process.

The challenge focuses on leveraging open-source tooling, multitask learning, and transfer learning to improve predictive performance on curated ADMET datasets.

Key Components
--------------

The framework is organized into several modular components, each designed to handle a specific part of the machine learning lifecycle:

* **Data Pipeline** (`admet.data`): Handles data ingestion, harmonization, and standardization. It also includes tools for generating molecular fingerprints and creating reproducible dataset splits.
* **Model Abstractions** (`admet.model`): Provides a unified interface for various machine learning models (e.g., XGBoost, LightGBM), making it easy to experiment with different algorithms.
* **Training Orchestration** (`admet.train`): Manages the training process, supporting both local execution and distributed training via Ray. It handles dataset loading, model initialization, and artifact persistence.
* **Evaluation & Visualization** (`admet.evaluate`, `admet.visualize`): Offers a suite of metrics and plotting utilities to assess model performance and explore dataset characteristics.

Getting Started
---------------

If you are new to the project, we recommend starting with the :doc:`cli` guide to learn how to use the command-line interface. For those interested in contributing or extending the framework, the :doc:`development` guide provides essential setup instructions.
