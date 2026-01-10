========
Glossary
========

Technical terms used throughout the OpenADMET Challenge documentation.

.. glossary::
   :sorted:

   ADMET
      Absorption, Distribution, Metabolism, Excretion, and Toxicity. The
      pharmacokinetic and safety properties that determine whether a drug
      candidate will be viable.

   ASHA
      Asynchronous Successive Halving Algorithm. A hyperparameter optimization
      scheduler that aggressively terminates poorly-performing trials to focus
      resources on promising configurations.

   BitBirch
      A molecular clustering algorithm that groups structurally similar
      molecules. Used for cluster-aware data splitting to prevent data leakage.

   Caco-2
      Human colon adenocarcinoma cell line used to model intestinal epithelium.
      Caco-2 permeability assays predict oral drug absorption.

   Chemprop
      Message-passing neural network (MPNN) architecture for molecular property
      prediction. Uses learned representations of molecular graphs rather than
      fixed fingerprints.

   Chemeleon
      Foundation model for molecular property prediction based on pretrained
      representations. Can be fine-tuned for specific endpoints.

   CLint
      Intrinsic clearance. The rate of drug metabolism by hepatic enzymes,
      measured in μL/min/mg protein.

   Curriculum Learning
      Training strategy that presents examples in order of difficulty or quality.
      Models first learn from clean/easy data, then progressively include
      noisier/harder examples.

   D-MPNN
      Directed Message-Passing Neural Network. The specific MPNN architecture
      used by Chemprop where messages flow along directed edges.

   Efflux
      Active transport of drugs out of cells, typically mediated by P-glycoprotein
      (P-gp). High efflux reduces drug absorption and CNS penetration.

   Ensemble
      Collection of multiple models whose predictions are combined (usually
      averaged) to improve accuracy and uncertainty quantification.

   FFN
      Feed-Forward Network. The fully-connected layers that process molecular
      representations to produce final predictions.

   Fold
      One iteration of cross-validation. In 5-fold CV, data is split into 5
      parts and each fold uses a different part for validation.

   HPO
      Hyperparameter Optimization. Systematic search for optimal model
      configuration (learning rate, architecture, regularization, etc.).

   LogD
      Distribution coefficient. The ratio of drug concentration in octanol
      vs water at physiological pH (7.4). Measures lipophilicity.

   MAE
      Mean Absolute Error. Average of absolute differences between predicted
      and actual values. Primary metric for regression tasks.

   MLflow
      Open-source platform for experiment tracking, model registry, and
      deployment. Logs metrics, parameters, and artifacts.

   MoE
      Mixture of Experts. FFN architecture where multiple expert networks
      specialize in different parts of the input space, with a gating
      network selecting which experts to use.

   MPNN
      Message-Passing Neural Network. Graph neural network that iteratively
      updates node representations by aggregating information from neighbors.

   Multi-task Learning
      Training a single model to predict multiple targets simultaneously.
      Shared representations can improve generalization when targets are related.

   Optuna
      Hyperparameter optimization framework with efficient sampling strategies.
      Supports warmstarting from previous studies.

   P-gp
      P-glycoprotein. An efflux transporter that pumps drugs out of cells,
      reducing absorption and CNS penetration.

   Papp
      Apparent permeability coefficient. Rate of drug transport across a
      cell monolayer, measured in cm/s.

   Ray Tune
      Distributed hyperparameter tuning library. Manages parallel trial
      execution across CPUs/GPUs with various schedulers (ASHA, PBT).

   RMSE
      Root Mean Square Error. Square root of average squared differences
      between predicted and actual values. Penalizes large errors more than MAE.

   SMILES
      Simplified Molecular-Input Line-Entry System. Text representation of
      molecular structure (e.g., ``CCO`` for ethanol).

   Split
      Division of data into train/validation/test sets. Cluster-aware splitting
      ensures similar molecules stay in the same split.

   Task Affinity
      Measure of gradient alignment between different prediction tasks.
      Tasks with positive affinity benefit from joint training; negative
      affinity suggests conflicts.

   TPE
      Tree-structured Parzen Estimator. Bayesian optimization algorithm used
      by Optuna to efficiently explore hyperparameter spaces.

   Trial
      One hyperparameter configuration evaluated during HPO. Each trial
      trains a model with specific settings and reports validation metrics.

   Warmstart
      Initializing optimization from previous results rather than from scratch.
      Warmstarting HPO loads top trials from a previous study to accelerate
      convergence.
