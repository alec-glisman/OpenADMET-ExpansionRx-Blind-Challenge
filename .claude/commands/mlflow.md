# MLflow Server Management

Start and manage MLflow tracking server for experiment logging.

## Start Server

```bash
# Default (port 8080)
mlflow ui --port 8080

# With specific backend store
mlflow server --backend-store-uri sqlite:///mlflow.db --port 8080

# Background process
nohup mlflow ui --port 8080 > mlflow.log 2>&1 &
```

## Access UI

Open http://127.0.0.1:8080 in browser.

## Config Integration

Set in YAML configs:
```yaml
mlflow:
  enabled: true
  tracking_uri: http://127.0.0.1:8080
  experiment_name: admet
  run_name: my_experiment
  log_model: true
  log_predictions: true
```

## View Experiments

```bash
# List experiments
mlflow experiments search

# View specific run
mlflow runs get --run-id <run_id>
```

## Disable MLflow

Set in config:
```yaml
mlflow:
  enabled: false
```

## Artifacts

Models and predictions logged to `mlruns/` directory or configured artifact store.
