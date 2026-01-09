# Create Config Files

Create new configuration files for training and HPO.

## Config Locations

- `configs/0-experiment/` - Single model experiments
- `configs/1-hpo-single-fold/` - HPO configs
- `configs/3-production/` - Production ensemble configs

## Template: Chemprop Training

```yaml
model:
  type: chemprop
  chemprop:
    depth: 5
    message_hidden_dim: 300
    dropout: 0.1
    ffn_type: regression  # regression, mixture_of_experts, branched
    num_layers: 2
    hidden_dim: 300
    batch_norm: true

optimization:
  learning_rate: 0.001
  max_epochs: 100
  patience: 10
  weight_decay: 0.0

data:
  data_dir: assets/dataset/splits/split_0/fold_0/
  smiles_col: SMILES
  target_cols:
    - LogD
    - Log KSOL
    - Log HLM CLint
    - Log MLM CLint
    - Log Caco-2 Permeability Papp A>B
    - Log Caco-2 Permeability Efflux
    - Log MPPB
    - Log MBPB
    - Log MGMB

mlflow:
  enabled: true
  tracking_uri: http://127.0.0.1:8080
  experiment_name: admet
```

## Template: HPO Config

See `configs/1-hpo-single-fold/hpo_chemprop.yaml` for full HPO template with search spaces.

## Validation

```bash
# Validate config loads correctly
python -c "from omegaconf import OmegaConf; OmegaConf.load('config.yaml')"
```
