# Train Single Model

Train a single ADMET model from a YAML config.

## Command

```bash
admet model train -c <config_path>
```

## Model Types

- `chemprop` - MPNN (message passing neural network)
- `chemeleon` - Pretrained encoder with FFN head
- `xgboost` - XGBoost with molecular fingerprints
- `lightgbm` - LightGBM with fingerprints
- `catboost` - CatBoost with fingerprints

## Example Configs

```bash
# Chemprop baseline
admet model train -c configs/0-experiment/chemprop.yaml

# Chemeleon (pretrained)
admet model train -c configs/0-experiment/chemeleon.yaml

# XGBoost
admet model train -c configs/4-more-models/xgboost.yaml
```

## Override Model Type

```bash
admet model train -c config.yaml --model-type chemprop
```

## ADMET Targets (9)

LogD, Log KSOL, Log HLM CLint, Log MLM CLint, Log Caco-2 Papp A>B, Log Caco-2 Efflux, Log MPPB, Log MBPB, Log MGMB

## Output

Models saved to `output_dir` specified in config. MLflow logs to configured tracking URI.
