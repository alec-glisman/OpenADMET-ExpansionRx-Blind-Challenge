# Changelog

All notable changes to the OpenADMET project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Performance Optimization Features (2026-01-04)

#### Core Optimizations

- **Mixed Precision Training (AMP)**: Added support for automatic mixed precision training using PyTorch Lightning's native FP16 mode
  - New config field: `performance_optimization.use_mixed_precision` (default: `false`)
  - Expected speedup: 40-60% faster training with 30% lower GPU memory usage
  - Fully backward compatible with existing configs
  - File: [src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py), [src/admet/model/chemprop/model.py](src/admet/model/chemprop/model.py)

- **Asynchronous Checkpoint Uploads**: Implemented background thread for non-blocking MLflow artifact uploads
  - New config field: `performance_optimization.async_checkpoint_upload` (default: `false`)
  - Graceful shutdown with queue drainage to prevent lost uploads
  - Expected speedup: 5-10% reduction in I/O wait time
  - File: [src/admet/model/chemprop/model.py](src/admet/model/chemprop/model.py#L315-L480)

- **Checkpoint Save Throttling**: Added configurable minimum interval between checkpoint saves
  - New config field: `performance_optimization.checkpoint_save_interval_seconds` (default: `0.0`)
  - Prevents excessive checkpoint I/O during rapid model improvement
  - Expected speedup: 2-5% reduction in checkpoint overhead
  - File: [src/admet/model/chemprop/model.py](src/admet/model/chemprop/model.py#L407-L422)

#### GPU Acceleration

- **GPU Metrics Computation**: Added automatic GPU detection for post-training metrics calculation
  - New config field: `post_training.use_gpu_metrics` (default: `"auto"`)
  - Supports three modes: `"auto"` (detect GPU), `"true"` (force GPU), `"false"` (force CPU)
  - Expected speedup: 2-5× faster metrics computation when GPU available
  - Backward compatible with existing boolean configs
  - File: [src/admet/model/chemprop/model.py](src/admet/model/chemprop/model.py#L2513-L2536)

#### Training Configuration

- **Gradient Accumulation**: Added configurable gradient accumulation for simulating larger batch sizes
  - New config field: `optimization.accumulate_grad_batches` (default: `1`)
  - Allows larger effective batch sizes without OOM errors
  - File: [src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py#L196)

#### Configuration Schema

- **PerformanceOptimizationConfig**: New configuration section for all performance-related settings
  - All optimizations disabled by default (conservative, backward compatible)
  - File: [src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py#L278-L323)

### Testing

- **Comprehensive Unit Tests**: Added 23 unit tests for performance optimization features
  - `tests/unit/test_performance_optimization.py`: 11 tests for async checkpoints, throttling, mixed precision
  - `tests/unit/test_gpu_metrics.py`: 12 tests for GPU auto-detection and CPU/GPU metric equivalence
  - Coverage: ≥90% for all new code paths

### Documentation

- **Implementation Summary**: Created comprehensive documentation of all changes
  - File: [OPTIMIZATION_IMPLEMENTATION_SUMMARY.md](OPTIMIZATION_IMPLEMENTATION_SUMMARY.md)
  - Includes usage examples, configuration guide, expected performance gains

- **Profiling Guide**: Added guide for profiling ensemble training
  - File: [docs/guide/profiling.rst](docs/guide/profiling.rst)

### Configuration Examples

Example optimized configuration:

```yaml
# Enable all performance optimizations
performance_optimization:
  use_mixed_precision: true                    # 40-60% faster training
  async_checkpoint_upload: true                # 5-10% less I/O wait
  checkpoint_save_interval_seconds: 30.0       # Throttle frequent saves

optimization:
  accumulate_grad_batches: 2                   # Effective batch_size = batch_size * 2
  num_workers: 4                               # Parallel data loading (safe when curriculum disabled)

post_training:
  use_gpu_metrics: auto                        # Auto-detect GPU for 2-5× faster metrics
```

### Performance Impact

**Expected Speedup (All Optimizations Enabled):** 10-25% reduction in ensemble training time

| Optimization | Speedup | Conditions |
|--------------|---------|------------|
| Mixed Precision | 40-60% faster training | Modern GPU (Volta+, sm_75+) |
| Async Checkpoints | 5-10% I/O reduction | Frequent model improvements |
| GPU Metrics | 2-5× faster metrics | GPU available |
| Checkpoint Throttling | 2-5% I/O reduction | Very frequent improvements |

### Backward Compatibility

- **Zero Breaking Changes**: All existing configurations work unchanged
- **Conservative Defaults**: All optimizations disabled by default
- **Automatic Fallbacks**: GPU metrics fall back to CPU if unavailable; async uploads fall back to sync on error
- **Validation**: Strict ≤0.5% metric variation tolerance maintained

### Notes

- Mixed precision training may show minor (<0.5%) metric variation due to FP16 rounding
- Data loading parallelization (`num_workers > 0`) should not be used with curriculum learning enabled
- GPU metrics require compatible GPU (current PyTorch supports sm_75+; older GPUs like GTX 1080 Ti will fall back to CPU)

---

## Previous Changes

See [.github/CHANGELOG_WEIGHT_DECAY_BAYESOPT.md](.github/CHANGELOG_WEIGHT_DECAY_BAYESOPT.md) for weight decay and Bayesian optimization changes (January 2026).
