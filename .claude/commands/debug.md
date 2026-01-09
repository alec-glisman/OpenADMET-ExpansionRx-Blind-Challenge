# Debug and Profile Training

Tools for troubleshooting and profiling model training.

## CUDA Memory Issues

```bash
# Check GPU memory
nvidia-smi

# Monitor continuously
watch -n 1 nvidia-smi

# Clear CUDA cache in Python
python -c "import torch; torch.cuda.empty_cache()"
```

## Profile Training

```python
# In training script, enable profiler
from admet.util.profiling import TrainingProfiler

profiler = TrainingProfiler()
# ... training code ...
profiler.report()
```

## Common Issues

### Out of Memory
- Reduce `batch_size` in config
- Reduce `message_hidden_dim` or `hidden_dim`
- Use gradient accumulation

### Ray Tune Failures
- Check `/tmp/ray/` for logs
- Set `RAY_DEDUP_LOGS=0` for verbose output
- Increase `max_concurrent_trials`

### MLflow Connection Errors
- Verify server running: `curl http://127.0.0.1:8080`
- Check `tracking_uri` in config
- Set `mlflow.enabled: false` to bypass

## Verbose Logging

```yaml
logging:
  enabled: true
  verbose: 2  # 0=quiet, 1=standard, 2=debug
```

## Test Single Trial

```bash
# Run HPO with 1 trial for debugging
admet model hpo -c config.yaml --num-samples 1
```
