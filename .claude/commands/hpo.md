# Hyperparameter Optimization

Run HPO with Ray Tune and Optuna Bayesian optimization.

## Command

```bash
admet model hpo -c <config_path> --num-samples <N>
```

## Example

```bash
# Chemprop HPO (50 trials)
admet model hpo -c configs/1-hpo-single-fold/hpo_chemprop.yaml --num-samples 50

# Chemeleon HPO
admet model hpo -c configs/1-hpo-single-fold/hpo_chemeleon.yaml --num-samples 50
```

## Multi-Phase HPO

Run the 3-phase workflow for best results:

```bash
# Phase 1: Exploration (wide search)
admet model hpo -c configs/1-hpo-single-fold/phases/phase1_explore_chemprop.yaml

# Phase 2: Exploitation (narrowed search, warmstart)
admet model hpo -c configs/1-hpo-single-fold/phases/phase2_exploit_chemprop.yaml

# Phase 3: Refinement (focused, final tuning)
admet model hpo -c configs/1-hpo-single-fold/phases/phase3_refine_chemprop.yaml
```

## Key Config Options

```yaml
search_algorithm:
  type: optuna  # optuna, bayesopt, hyperopt, random
  n_initial_points: 20

asha:
  grace_period: 15
  reduction_factor: 3
```

## Results

- Best configs: `hpo_results/top_k_configs.json`
- Studies: `hpo_results/optuna_studies/studies.db`
- MLflow: Check tracking URI for metrics
