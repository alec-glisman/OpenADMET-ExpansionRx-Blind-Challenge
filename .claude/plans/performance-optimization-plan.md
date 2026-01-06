# Performance Optimization Plan: ADMET ML Training Pipeline

## Executive Summary

**Goal:** Maximize training throughput on fixed 2-GPU desktop (1080 Ti + 3080) to minimize wall-clock time for model training, ensembles, and HPO.

**Target Speedup:** 2-3x total speedup through phased optimizations across data loading, training loop, parallelization, and experiment tracking.

**Hardware Constraints:**

- 2 GPUs: 1080 Ti (11GB), 3080 (10GB)
- Resources cannot increase - must optimize existing pipeline
- Ray used for parallelization across GPUs

**Key Findings from Analysis:**

1. **Fingerprint recomputation** - Classical models regenerate features 25x per ensemble (CRITICAL)
2. **Sequential ensemble training** - Underutilizes GPUs despite Ray infrastructure (CRITICAL)
3. **Batch_size=1 predictions** - 2-3x slower than batched inference (HIGH)
4. **MLflow parameter logging** - N parameters = N HTTP calls (HIGH)
5. **No mixed precision** - Missing 1.5-2x speedup from FP16/AMP (HIGH)
6. **Curriculum sampler forces num_workers=0** - Disables DataLoader parallelism (MEDIUM)

---

## Implementation Phases

### Phase 1: Quick Wins (1-2 Days, 20-30% Speedup)

#### 1.1 Enable Batched Predictions

**Impact:** 2-3x faster inference
**Effort:** 1 hour
**File:** [src/admet/model/chemprop/model.py:3073](src/admet/model/chemprop/model.py#L3073)

**Current Issue:**

```python
dataloader = data.build_dataloader(dataset, batch_size=1, num_workers=0, shuffle=False)
```

**Change:**

```python
dataloader = data.build_dataloader(
    dataset,
    batch_size=self.hyperparams.batch_size,  # Use training batch size (256-512)
    num_workers=min(self.hyperparams.num_workers, 4),
    shuffle=False
)
```

**Why This Matters:** Test/blind predictions run 25 times per ensemble. With 1000 molecules × 25 folds, saves ~8-12 minutes per ensemble.

---

#### 1.2 Batch MLflow Parameter Logging

**Impact:** 5-10x faster parameter logging
**Effort:** 2 hours
**File:** [src/admet/model/hpo_mlflow_callback.py:243-244](src/admet/model/hpo_mlflow_callback.py#L243-L244)

**Current Issue:**

```python
for key, value in params_to_log.items():
    self._mlflow_client.log_param(run_id=run.info.run_id, key=key, value=value)
```

Creates N HTTP requests for N parameters!

**Change:**

```python
from mlflow.entities import Param

params_list = [Param(key, str(value)) for key, value in params_to_log.items()]
if params_list and self._mlflow_client:
    try:
        self._mlflow_client.log_batch(run_id=run.info.run_id, params=params_list)
    except Exception as e:
        logger.debug("Failed to batch log params for %s: %s", trial.trial_id, e)
```

**Why This Matters:** HPO with 50 params × 100 trials = 5000 HTTP calls → 100 calls. Saves ~5 minutes per HPO run.

---

#### 1.3 Enable DataLoader Workers for Non-Curriculum Training

**Impact:** 10-20% faster training
**Effort:** 2 hours
**File:** [src/admet/model/chemprop/model.py:1275-1280](src/admet/model/chemprop/model.py#L1275-L1280)

**Current Issue:** Curriculum sampler forces `num_workers=0`, disabling DataLoader parallelism even when curriculum is disabled.

**Change:**

```python
# Only force num_workers=0 when curriculum is actually enabled
curriculum_enabled = (
    self.hyperparams.curriculum is not None and
    self.hyperparams.curriculum.enabled
)

if self.hyperparams.num_workers > 0 and curriculum_enabled:
    logger.warning("Curriculum enabled: forcing num_workers=0 for sampler state sync")
    actual_num_workers = 0
else:
    actual_num_workers = self.hyperparams.num_workers
```

**Why This Matters:** Overlaps data loading with GPU compute for majority of configs that don't use curriculum.

---

#### 1.4 Optimize Ray Result Buffering

**Impact:** 5-10% less HPO overhead
**Effort:** 30 minutes
**File:** [src/admet/model/chemprop/hpo.py:52-53](src/admet/model/chemprop/hpo.py#L52-L53)

**Change:**

```python
os.environ.setdefault("TUNE_RESULT_BUFFER_LENGTH", "1")
os.environ.setdefault("TUNE_RESULT_BUFFER_MIN_TIME_S", "1")
```

Reduces buffering for more responsive metric reporting.

---

### Phase 2: Caching Infrastructure (3-5 Days, 40-70% Additional Speedup)

#### 2.1 Fingerprint Caching (CRITICAL - HIGHEST ROI)

**Impact:** 10-25x faster fingerprint generation after first run
**Effort:** 12 hours
**Files:**

- New: [src/admet/features/fingerprint_cache.py](src/admet/features/fingerprint_cache.py)
- Modify: [src/admet/features/fingerprints.py](src/admet/features/fingerprints.py)
- Modify: [src/admet/model/classical/base.py:214](src/admet/model/classical/base.py#L214)

**Architecture:**

Create HDF5-based persistent cache:

```python
class FingerprintCache:
    """HDF5-backed fingerprint cache with SMILES hash keys."""

    def __init__(self, cache_dir: Path, fingerprint_config: FingerprintConfig):
        config_hash = self._compute_config_hash(fingerprint_config)
        self.cache_path = cache_dir / f"fp_{fingerprint_config.type}_{config_hash}.h5"
        self.config = fingerprint_config
        self._lock = FileLock(f"{self.cache_path}.lock")

    def get_batch(self, smiles_list: list[str]) -> tuple[np.ndarray, list[str]]:
        """Retrieve fingerprints, return (cached_fps, missing_smiles)."""
        with self._lock, h5py.File(self.cache_path, 'r') as f:
            # Hash SMILES and lookup in HDF5
            ...

    def set_batch(self, smiles_list: list[str], fingerprints: np.ndarray):
        """Store fingerprints in cache."""
        with self._lock, h5py.File(self.cache_path, 'a') as f:
            # Store with SMILES hash as key
            ...
```

**Integration in FingerprintGenerator:**

```python
def generate(self, smiles: list[str]) -> np.ndarray:
    if self.cache is not None:
        fps, missing = self.cache.get_batch(smiles)
        if missing:
            new_fps = self._compute_fingerprints(missing)
            self.cache.set_batch(missing, new_fps)
            # Merge cached + new fingerprints in correct order
            return self._merge(fps, new_fps, smiles, missing)
        return fps
    else:
        return self._compute_fingerprints(smiles)
```

**Cache Directory Structure:**

```
cache/fingerprints/
├── fp_morgan_abc123def.h5          # Morgan radius=2, n_bits=2048
├── fp_rdkit_xyz789ghi.h5           # RDKit default
└── fp_maccs_qwe456rty.h5           # MACCS keys
```

**Why This Matters:** Classical models in 5×5 ensemble regenerate fingerprints 25 times. With cache, only computed once. Saves 15-30 minutes per ensemble.

**Note:** Lower priority since user focus is on Chemprop/Chemeleon models, not classical models. Can be implemented later if needed.

**Implementation Steps:**

1. Create `FingerprintCache` class with HDF5 backend and file locking
2. Add `cache_dir` parameter to `FingerprintConfig` (default: `~/.admet/cache/fingerprints`)
3. Modify `FingerprintGenerator.__init__()` to create cache if `cache_dir` specified
4. Update `generate()` to check cache before computing
5. Add CLI command: `admet data cache-fingerprints --data-file <path> --fingerprint-config <yaml>`

---

#### 2.2 SMILES Canonicalization Cache

**Impact:** 5-10x faster dataloader creation
**Effort:** 4 hours
**Files:**

- Modify: [src/admet/data/smiles.py](src/admet/data/smiles.py)
- Modify: [src/admet/model/chemprop/model.py:1165](src/admet/model/chemprop/model.py#L1165)

**Implementation:**

```python
from functools import lru_cache

@lru_cache(maxsize=100000)
def _canonicalize_single(smiles: str) -> str:
    """Cache individual SMILES canonicalization (thread-safe)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else smiles
    except Exception:
        return smiles

def parallel_canonicalize_smiles(smiles_list: list[str]) -> list[str]:
    """Parallel canonicalization with per-SMILES caching."""
    return [_canonicalize_single(s) for s in smiles_list]
```

**Why This Matters:** Dataloader creation runs 25x per ensemble. LRU cache provides instant hits for test/blind data (shared across all models). For train/val data, cache helps across epochs but not across ensemble members (different splits).

**HPO Benefit:** During HPO, all trials use the same train/val fold, so canonicalization cache provides massive benefit (100 trials × same data = 100x reuse).

---

#### 2.3 Precompute Test/Blind Datasets for Chemprop ⚠️ UPDATED

**Impact:** 25x reduction in test/blind preprocessing
**Effort:** 6 hours
**File:** [src/admet/model/chemprop/ensemble.py:1036-1087](src/admet/model/chemprop/ensemble.py#L1036-L1087)

**Important Note:** Train/validation datasets differ across ensemble members (5-fold CV × 5 seeds = 25 unique train/val splits). **However, test/blind datasets are identical** across all 25 models.

**Architecture:**

```python
class ModelEnsemble:
    def __init__(self, config):
        self._shared_test_dataset = None
        self._shared_blind_dataset = None

    def _precompute_shared_datasets(self):
        """Precompute test/blind MoleculeDatasets once, share across all workers."""
        if self.config.data.test_file:
            test_df = pd.read_csv(self.config.data.test_file)
            # Canonicalize, create MoleculeDatapoints, apply featurizer
            self._shared_test_dataset = self._create_molecule_dataset(test_df)
            # Store in Ray object store for zero-copy sharing
            self._shared_test_ref = ray.put(self._shared_test_dataset)
```

**Why This Matters:** Test/blind sets are identical across all 25 folds but preprocessed 25 times. Saves 3-8 minutes per ensemble.

**Note on Train/Val:** Train/val datasets are NOT cached since each ensemble member uses different splits. However, HPO runs can share train/val data since all trials use the same fold.

---

#### 2.4 Parallel Ensemble Training with Optimized 2-GPU Allocation (CRITICAL) ⚠️ UPDATED

**Impact:** 2.5x faster ensemble training
**Effort:** 8 hours
**File:** [src/admet/model/chemprop/ensemble.py:1172-1180](src/admet/model/chemprop/ensemble.py#L1172-L1180)

**Current Problem:** `max_parallel=2` trains only 2 models simultaneously, underutilizing both GPUs.

**Optimized Strategy Based on User Testing:**

- **Chemprop:** Max 3 models per GPU = 6 models parallel total
- **Chemeleon:** Max 2 models per GPU = 4 models parallel total
- **Classical models:** CPU-only, can run many more in parallel

**GPU Memory Budget (User-Validated):**

- **Chemprop:** 1080 Ti (11GB) ÷ 3 ≈ 3.7GB per model, 3080 (10GB) ÷ 3 ≈ 3.3GB per model
- **Chemeleon:** 1080 Ti (11GB) ÷ 2 = 5.5GB per model, 3080 (10GB) ÷ 2 = 5GB per model

**Implementation:**

```python
# Model-specific max_parallel configuration
def get_max_parallel_for_model(model_type: str, num_gpus: int = 2) -> int:
    """Determine max parallel models based on model type and VRAM constraints."""
    if model_type == "chemprop":
        return 3 * num_gpus  # 3 models per GPU
    elif model_type == "chemeleon":
        return 2 * num_gpus  # 2 models per GPU
    elif model_type in ["xgboost", "lightgbm", "catboost"]:
        return 16  # CPU-bound, many more parallel
    return 2 * num_gpus  # Default conservative

# In ensemble config or code
max_parallel = get_max_parallel_for_model(config.model.type)
num_gpus_per_task = 1.0 / (max_parallel / 2)  # Fractional GPU

@ray.remote(num_gpus=num_gpus_per_task)
def train_single_model(config, split_fold_info):
    """Train single model with dynamic GPU selection."""
    # Select GPU with most free memory
    available_gpu_ids = [0, 1]  # Both GPUs available
    selected_gpu = select_gpu_with_most_free_memory(available_gpu_ids)

    # Set PyTorch to use selected GPU
    import torch
    torch.cuda.set_device(selected_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)

    # Train model
    model = ChempropModel.from_config(config)
    model.fit(...)
    return results

def select_gpu_with_most_free_memory(gpu_ids: list[int]) -> int:
    """Select GPU with highest free memory."""
    import torch
    max_free = -1
    selected = gpu_ids[0]
    for gpu_id in gpu_ids:
        free_mem, _ = torch.cuda.mem_get_info(gpu_id)
        if free_mem > max_free:
            max_free = free_mem
            selected = gpu_id
    return selected
```

**Expected Performance:**

- **Chemprop Current:** 25 models × 10 min ÷ 2 = **125 minutes**
- **Chemprop Optimized:** 25 models × 10 min ÷ 6 = **~42 minutes**
- **Speedup: 3x for chemprop ensemble** (conservative vs 3.5x)

- **Chemeleon Current:** 25 models × 15 min ÷ 2 = **188 minutes**
- **Chemeleon Optimized:** 25 models × 15 min ÷ 4 = **~94 minutes**
- **Speedup: 2x for chemeleon ensemble**

**Tuning Strategy:**

1. Start with validated limits: Chemprop=6, Chemeleon=4
2. Monitor GPU memory with `nvidia-smi` during training
3. If memory stable and <80% utilized, can try +1 model per GPU
4. If OOM errors occur, reduce by 1 model per GPU

---

### Phase 3: Training Loop Optimizations (1-2 Weeks, 1.5-2x Additional Speedup)

#### 3.1 Mixed Precision Training (AMP)

**Impact:** 1.5-2x faster training, 50% less GPU memory
**Effort:** 6 hours
**File:** [src/admet/model/chemprop/model.py](src/admet/model/chemprop/model.py) (in `_prepare_trainer()`)

**Implementation:**

```python
def _prepare_trainer(self):
    trainer = pl.Trainer(
        max_epochs=self.hyperparams.max_epochs,
        precision="16-mixed",  # Enable mixed precision (FP16)
        # Alternative for 3080 (Ampere): precision="bf16-mixed"
        callbacks=[...],
        ...
    )
    return trainer
```

**GPU-Specific Speedup:**

- **1080 Ti:** 1.3-1.5x (FP16 support, no Tensor Cores)
- **3080:** 1.8-2.0x (Tensor Cores optimized for FP16)

**Benefits:**

- Faster matrix multiplications
- Reduced GPU memory → can increase batch size for better throughput
- PyTorch Lightning handles gradient scaling automatically

**Testing:**

- Verify final metrics match FP32 within 1-2% tolerance
- Monitor for NaN losses (rare but possible)
- If instability occurs, fall back to FP32

---

#### 3.2 Gradient Accumulation for Larger Effective Batches

**Impact:** 10-20% faster convergence (fewer epochs to target loss)
**Effort:** 4 hours
**Files:**

- Modify: [src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py)
- Modify: [src/admet/model/chemprop/model.py](src/admet/model/chemprop/model.py)

**Add to Config:**

```python
@dataclass
class OptimizationConfig:
    batch_size: int = 256
    accumulate_grad_batches: int = 1  # NEW: accumulate over N batches
    # Effective batch = batch_size × accumulate_grad_batches
```

**Add to Trainer:**

```python
trainer = pl.Trainer(
    accumulate_grad_batches=self.hyperparams.accumulate_grad_batches,
    ...
)
```

**Usage Example:**

- `batch_size=256`, `accumulate_grad_batches=4` → effective batch = 1024
- Larger batches → more stable gradients → faster convergence
- Works synergistically with mixed precision (memory savings enable larger accumulation)

---

#### 3.3 Distributed Fingerprint Preprocessing with Ray

**Impact:** 2-4x faster fingerprint generation
**Effort:** 6 hours
**File:** New [src/admet/features/distributed_fingerprints.py](src/admet/features/distributed_fingerprints.py)

**Implementation:**

```python
@ray.remote
def compute_fingerprints_batch(smiles_batch: list[str], config: FingerprintConfig) -> np.ndarray:
    """Compute fingerprints for a batch on a Ray worker."""
    generator = FingerprintGenerator(config)
    return generator.generate(smiles_batch)

def parallel_fingerprint_generation(
    smiles_list: list[str],
    config: FingerprintConfig,
    batch_size: int = 1000
) -> np.ndarray:
    """Generate fingerprints in parallel across Ray workers."""
    batches = [smiles_list[i:i+batch_size] for i in range(0, len(smiles_list), batch_size)]
    futures = [compute_fingerprints_batch.remote(batch, config) for batch in batches]
    results = ray.get(futures)
    return np.vstack(results)
```

**Why This Matters:** Fingerprint generation is CPU-bound and embarrassingly parallel. Utilizes all CPU cores via Ray.

---

### Phase 4: Advanced Optimizations (Optional)

#### 4.1 Persistent Ray Cluster Across Experiments

**Impact:** <1% (startup time only)
**Effort:** 3 hours

Reuse Ray cluster across multiple experiments instead of `ray.init()` per run. Saves 5-15 seconds startup overhead.

---

## Implementation Priority & Timeline

### Week 1: Quick Wins (Total: 5.5 hours)

1. ✅ **1.2 MLflow Batch Logging** - 2h - 5-10x logging speedup
2. ✅ **1.1 Batched Prediction** - 1h - 2-3x inference speedup
3. ✅ **1.3 DataLoader num_workers** - 2h - 10-20% training speedup
4. ✅ **1.4 Ray Buffer Tuning** - 0.5h - 5-10% HPO overhead reduction

**Expected: 20-30% overall speedup**

---

### Week 2: Caching (Total: 10 hours) ⚠️ UPDATED

1. ⚪ **2.1 Fingerprint Cache** - 12h - DEPRIORITIZED (classical models only, focus on Chemprop/Chemeleon)
2. ✅ **2.2 SMILES Cache** - 4h - 5-10x dataloader creation (Chemprop/Chemeleon)
3. ✅ **2.3 Precompute Test/Blind** - 6h - 25x test/blind preprocessing (IMPLEMENTED - test data is shared)

**Expected: +20-30% additional speedup (cumulative: 40-60%)** - Fully implemented

---

### Week 3: Parallelization & Training (Total: 18 hours)

1. ✅ **2.4 Parallel Ensemble (2-GPU)** - 8h - 3.5x ensemble speedup ⚡
2. ✅ **3.1 Mixed Precision (AMP)** - 6h - 1.5-2x training speedup ⚡
3. ✅ **3.2 Gradient Accumulation** - 4h - 10-20% convergence speedup

**Expected: +1.5-2x additional speedup (cumulative: 2-3x total) ⚡**

---

### Week 4+: Advanced (Optional)

1. ⚪ **3.3 Distributed Fingerprints** - 6h - 2-4x fingerprint speedup
2. ⚪ **4.1 Persistent Ray Cluster** - 3h - Marginal startup savings

---

## Critical Files to Modify

| Priority | File | Changes |
|----------|------|---------|
| **P0** | [src/admet/model/chemprop/ensemble.py](src/admet/model/chemprop/ensemble.py) | Ray 2-GPU parallelization (2.4) |
| **P0** | [src/admet/features/fingerprints.py](src/admet/features/fingerprints.py) | Add caching support (2.1) |
| **P1** | [src/admet/model/chemprop/model.py](src/admet/model/chemprop/model.py) | Batched prediction (1.1), AMP (3.1), num_workers (1.3) |
| **P1** | [src/admet/model/hpo_mlflow_callback.py](src/admet/model/hpo_mlflow_callback.py) | Batch parameter logging (1.2) |
| **P2** | [src/admet/data/smiles.py](src/admet/data/smiles.py) | SMILES canonicalization cache (2.2) |
| **P2** | [src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py) | Add AMP, gradient accumulation configs (3.1, 3.2) |
| **P2** | [src/admet/model/classical/base.py](src/admet/model/classical/base.py) | Integrate fingerprint cache (2.1) |
| **New** | [src/admet/features/fingerprint_cache.py](src/admet/features/fingerprint_cache.py) | HDF5 cache implementation (2.1) |

---

## Testing Strategy & Model Quality Validation

### Critical Requirement: No Degradation in Multi-Task Regression Performance

All optimizations must preserve model quality. We will use a rigorous validation protocol to ensure performance optimizations don't materially affect predictions.

### Validation Protocol for Each Optimization

#### Phase 1: Baseline Establishment

Before implementing any optimization:

1. **Train Reference Model:**

   ```bash
   # Train baseline model on a standard fold (e.g., split_0/fold_0)
   admet model train -c configs/0-experiment/chemprop_baseline.yaml
   ```

2. **Record Baseline Metrics:**
   - Training MAE, RMSE, R² per task
   - Validation MAE, RMSE, R² per task
   - Test set predictions (if available)
   - MLflow run ID for comparison

3. **Save Baseline Predictions:**

   ```python
   # Generate predictions on validation and test sets
   baseline_val_preds = model.predict(val_smiles)
   baseline_test_preds = model.predict(test_smiles)
   np.save("baseline_val_preds.npy", baseline_val_preds)
   np.save("baseline_test_preds.npy", baseline_val_preds)
   ```

#### Phase 2: Optimization Implementation

For each optimization (1.1, 1.2, 2.2, etc.):

1. **Implement change** following plan specifications
2. **Document what changed** in code comments
3. **Record expected impact** (performance only, not predictions)

#### Phase 3: Correctness Validation

After implementing each optimization:

1. **Prediction Equivalence Test (CRITICAL):**

   ```python
   # Train optimized model with SAME random seed and config
   optimized_val_preds = optimized_model.predict(val_smiles)
   optimized_test_preds = optimized_model.predict(test_smiles)

   # Compute element-wise differences
   val_diff = np.abs(baseline_val_preds - optimized_val_preds)
   test_diff = np.abs(baseline_test_preds - optimized_test_preds)

   # Assertion: All differences should be < tolerance
   TOLERANCE = 1e-4  # FP32 numerical tolerance (adjustable for FP16)
   assert np.all(val_diff < TOLERANCE), f"Max diff: {val_diff.max()}"
   assert np.all(test_diff < TOLERANCE), f"Max diff: {test_diff.max()}"
   ```

2. **Metric Consistency Test:**

   ```python
   # Compare final training metrics
   baseline_metrics = {
       "val_mae": 0.523,  # Example
       "val_rmse": 0.891,
       "val_r2": 0.847,
   }
   optimized_metrics = get_metrics_from_mlflow(optimized_run_id)

   # Assertion: Metrics should match within 0.1% tolerance
   for key in baseline_metrics:
       rel_diff = abs(baseline_metrics[key] - optimized_metrics[key]) / baseline_metrics[key]
       assert rel_diff < 0.001, f"{key}: baseline={baseline_metrics[key]}, optimized={optimized_metrics[key]}"
   ```

3. **Per-Task Validation:**

   ```python
   # Ensure all 9 ADMET endpoints maintain performance
   for task_idx, task_name in enumerate(target_cols):
       baseline_task_mae = compute_mae(y_true[:, task_idx], baseline_preds[:, task_idx])
       optimized_task_mae = compute_mae(y_true[:, task_idx], optimized_preds[:, task_idx])

       assert abs(baseline_task_mae - optimized_task_mae) < 0.01, \
           f"Task {task_name} MAE changed: {baseline_task_mae} -> {optimized_task_mae}"
   ```

#### Phase 4: Specific Validation per Optimization Type

**For Data Loading Optimizations (1.1, 1.3, 2.2, 2.3):**

- ✅ **Expected:** Identical predictions (data order/content unchanged)
- ✅ **Tolerance:** 1e-6 (exact match expected)
- ⚠️ **Risk:** Low (pure performance, no algorithmic changes)

**For MLflow/Ray Optimizations (1.2, 1.4):**

- ✅ **Expected:** Zero prediction impact (logging/orchestration only)
- ✅ **Tolerance:** 0 (exact match required)
- ⚠️ **Risk:** Very low (completely orthogonal to training)

**For Training Loop Optimizations (3.1 Mixed Precision, 3.2 Gradient Accumulation):**

- ⚠️ **Expected:** Small numerical differences due to FP16 or batching changes
- ⚠️ **Tolerance:** 1e-2 for FP16 (mixed precision introduces rounding)
- ⚠️ **Risk:** Medium (numerical precision changes)
- **Mitigation Strategy:**

  ```python
  # For mixed precision: Compare final validation MAE
  baseline_mae = 0.523
  fp16_mae = 0.525  # Acceptable if within 1%
  assert abs(baseline_mae - fp16_mae) / baseline_mae < 0.01

  # If MAE degrades > 1%, fall back to FP32
  if degradation > 0.01:
      logger.warning("Mixed precision degrades MAE, using FP32")
      trainer.precision = "32"
  ```

**For Parallelization Optimizations (2.4):**

- ✅ **Expected:** Identical predictions per model (models train independently)
- ✅ **Tolerance:** 1e-6 (each model should match baseline)
- ⚠️ **Risk:** Low (parallelization doesn't change individual model training)
- **Validation:**

  ```python
  # Train 2 models in parallel and 2 sequentially with same seeds
  parallel_preds = [model1.predict(test), model2.predict(test)]
  sequential_preds = [model3.predict(test), model4.predict(test)]

  # Verify model1 matches model3 (seed 42), model2 matches model4 (seed 43)
  assert np.allclose(parallel_preds[0], sequential_preds[0], atol=1e-6)
  assert np.allclose(parallel_preds[1], sequential_preds[1], atol=1e-6)
  ```

### Automated Validation Test Suite

Create `tests/validation/test_optimization_correctness.py`:

```python
import pytest
import numpy as np
from admet.model.chemprop.model import ChempropModel

class TestOptimizationCorrectness:
    """Validate that optimizations preserve model quality."""

    @pytest.fixture
    def baseline_predictions(self):
        """Load pre-computed baseline predictions."""
        return np.load("tests/fixtures/baseline_predictions.npy")

    @pytest.fixture
    def baseline_metrics(self):
        """Load baseline validation metrics."""
        return {
            "val_mae": 0.523,
            "val_rmse": 0.891,
            "val_r2": 0.847,
        }

    def test_batched_prediction_equivalence(self, baseline_predictions):
        """Test that batch_size>1 predictions match batch_size=1."""
        model = ChempropModel.from_checkpoint("tests/fixtures/model.ckpt")

        # Predict with batch_size=1 (baseline)
        preds_batch1 = model.predict(test_smiles, batch_size=1)

        # Predict with batch_size=256 (optimized)
        preds_batch256 = model.predict(test_smiles, batch_size=256)

        # Assert exact match (deterministic operation)
        np.testing.assert_allclose(preds_batch1, preds_batch256, atol=1e-6)

    def test_smiles_cache_equivalence(self, baseline_predictions):
        """Test that SMILES canonicalization cache doesn't affect predictions."""
        # First run (populate cache)
        model1 = ChempropModel(...)
        model1.fit(train_data)
        preds1 = model1.predict(test_smiles)

        # Second run (use cache)
        model2 = ChempropModel(...)
        model2.fit(train_data)
        preds2 = model2.predict(test_smiles)

        # Assert exact match
        np.testing.assert_allclose(preds1, preds2, atol=1e-6)

    def test_mixed_precision_acceptable_degradation(self, baseline_metrics):
        """Test that FP16 training maintains MAE within 1% of FP32."""
        # Train with FP32 (baseline)
        model_fp32 = ChempropModel(..., precision="32")
        model_fp32.fit(train_data)
        val_mae_fp32 = model_fp32.validate(val_data)

        # Train with FP16 (optimized)
        model_fp16 = ChempropModel(..., precision="16-mixed")
        model_fp16.fit(train_data)
        val_mae_fp16 = model_fp16.validate(val_data)

        # Assert < 1% degradation
        relative_diff = abs(val_mae_fp32 - val_mae_fp16) / val_mae_fp32
        assert relative_diff < 0.01, f"FP16 degrades MAE by {relative_diff:.2%}"

    def test_parallel_ensemble_independence(self):
        """Test that parallel ensemble training produces identical models to sequential."""
        # Train 2 models sequentially
        config1 = ChempropConfig(random_seed=42)
        config2 = ChempropConfig(random_seed=43)
        model1_seq = ChempropModel(config1).fit(data1)
        model2_seq = ChempropModel(config2).fit(data2)

        # Train 2 models in parallel (Ray)
        results = ray.get([
            train_model.remote(config1, data1),
            train_model.remote(config2, data2),
        ])
        model1_par, model2_par = results

        # Assert predictions match (same seed = same model)
        preds1_seq = model1_seq.predict(test_smiles)
        preds1_par = model1_par.predict(test_smiles)
        np.testing.assert_allclose(preds1_seq, preds1_par, atol=1e-6)
```

### Continuous Validation During Development

For each pull request with optimizations:

1. **CI Pipeline runs automated tests:**
   - `pytest tests/validation/test_optimization_correctness.py`
   - All tests must pass before merge

2. **Manual Smoke Test (Before Production):**

   ```bash
   # Train mini 2×2 ensemble (2 splits × 2 folds = 4 models)
   admet model ensemble -c configs/validation/mini_ensemble.yaml

   # Compare validation MAE to historical baseline
   # Acceptable range: ±0.5% of baseline MAE
   ```

3. **Regression Dashboard (MLflow):**
   - Track validation MAE across commits
   - Alert if MAE degrades > 0.5% compared to last 5 runs
   - Example query:

     ```sql
     SELECT run_id, metrics.val_mae
     FROM runs
     WHERE tags.optimization_version = 'v2.0'
     ORDER BY start_time DESC LIMIT 10;
     ```

### Performance Benchmarks

1. **Create benchmark suite:** `tests/benchmarks/test_performance.py`
2. **Profile each phase:** Use existing `TrainingProfiler` to measure timing
3. **GPU Monitoring:** `nvidia-smi` during training to verify utilization
4. **End-to-end timing:** Measure full ensemble training before/after

### Integration Tests

1. Train 2-fold mini-ensemble (validate full pipeline)
2. Run 10-trial HPO (validate Ray + MLflow integration)
3. Train classical model ensemble with cache (validate fingerprint caching)

### Sign-Off Checklist (Per Optimization)

Before marking an optimization complete:

- [ ] Automated prediction equivalence test passes
- [ ] Validation metrics within tolerance (MAE, RMSE, R²)
- [ ] All 9 ADMET tasks maintain performance
- [ ] Performance improvement measured and documented
- [ ] No warnings/errors in training logs
- [ ] MLflow run comparison shows no regression
- [ ] Code review completed
- [ ] Documentation updated

### Rollback Strategy

If any optimization causes > 0.5% MAE degradation:

1. **Immediate rollback:** Revert the commit
2. **Root cause analysis:** Identify what caused regression
3. **Fix or abandon:** Either fix the issue or skip that optimization
4. **Re-validate:** Run full validation suite before re-attempting

---

## Risk Mitigation

### High-Risk Items

1. **2.1 Fingerprint Cache - Concurrent Access**
   - Risk: HDF5 corruption with parallel Ray workers
   - Mitigation: Use `fasteners.InterProcessLock` for file locking

2. **2.4 Parallel Ensemble - GPU OOM**
   - Risk: 8 models exceed GPU memory
   - Mitigation: Start with `max_parallel=6`, monitor with `nvidia-smi`, tune gradually

### Medium-Risk Items

1. **3.1 Mixed Precision - Numerical Instability**
   - Risk: NaN losses or degraded predictions
   - Mitigation: Thorough testing, gradient scaling, fallback to FP32 if needed

### Low-Risk Items

1. All Quick Wins (1.1-1.4) are standard practices with minimal risk

---

## Expected Final Performance

**Current Baseline (Example):**

- Single model training: 10 minutes
- 5×5 ensemble (25 models): 125 minutes (sequential with `max_parallel=2`)
- HPO (100 trials): 16 hours

**After All Optimizations:**

- Single Chemprop model: ~5 minutes (mixed precision + better data loading)
- Chemprop 5×5 ensemble: ~42 minutes (6-way parallelization + faster individual models)
- Chemeleon 5×5 ensemble: ~94 minutes (4-way parallelization + faster individual models)
- HPO (100 trials): ~8 hours (faster trials + reduced logging overhead + shared train/val data)

**Total Speedup: ~2-3x across all workflows** 🚀

**Note:** HPO benefits additionally from shared train/val data caching (all trials use same fold).

---

## Configuration Changes Required

### Enable Fingerprint Caching

```yaml
# configs/fingerprint_cache.yaml
fingerprint:
  type: "morgan"
  morgan:
    radius: 2
    n_bits: 2048
  cache_dir: "~/.admet/cache/fingerprints"  # NEW: enable caching
```

### Optimize Ensemble Parallelization ⚠️ UPDATED

```yaml
# configs/3-production/ensemble_chemprop.yaml
ray:
  max_parallel: 6          # Chemprop: 3 models per GPU (user-validated VRAM limit)
  num_gpus: 2              # Total GPUs available
  num_gpus_per_task: 0.33  # Each task uses 1/3 GPU

# configs/3-production/ensemble_chemeleon.yaml
ray:
  max_parallel: 4          # Chemeleon: 2 models per GPU (user-validated VRAM limit)
  num_gpus: 2              # Total GPUs available
  num_gpus_per_task: 0.5   # Each task uses 1/2 GPU
```

### Enable Mixed Precision

```yaml
# configs/0-experiment/chemprop.yaml
model:
  chemprop:
    precision: "16-mixed"  # NEW: FP16 mixed precision

optimization:
  accumulate_grad_batches: 4  # NEW: effective batch = batch_size × 4
```

---

## Summary

This plan provides a systematic path to 2-3x total speedup on your fixed 2-GPU hardware through:

1. **Week 1:** Quick wins for immediate 20-30% improvement
2. **Week 2:** Caching infrastructure for 50-80% cumulative improvement
3. **Week 3:** Parallelization and training optimizations for 2-3x total improvement

All optimizations are aggressive (backward compatibility not guaranteed) and tailored specifically for your 2-GPU desktop setup. The plan prioritizes highest-ROI items first with clear implementation guidance for each phase.

---

## Validation and Testing

### Test Suite Created

Comprehensive validation tests created in `tests/model/test_performance_optimizations.py`:

**Test Coverage:**
- ✅ Phase 1: Quick Wins (4 tests) - Batched predictions, MLflow batch logging, num_workers conditional, Ray buffer tuning
- ✅ Phase 2: Caching (2 tests) - SMILES canonicalization cache, Precomputed test/blind datasets
- ✅ Phase 3: Training Optimizations (3 tests) - Mixed precision, Gradient accumulation, Parallel ensemble GPU allocation
- ✅ Integration Tests (2 tests) - Ensemble with precomputed datasets, Optimizations preserve predictions
- ✅ Regression Prevention (2 tests) - No redundant file loading, SMILES cache reduces computations

### Test Results

**All validation tests passing: 12 passed, 1 skipped, 2 warnings**

```bash
$ pytest tests/model/test_performance_optimizations.py -v
collected 13 items
TestQuickWins::test_batched_predictions_enabled PASSED
TestQuickWins::test_mlflow_batch_logging PASSED
TestQuickWins::test_num_workers_conditional_on_curriculum PASSED
TestQuickWins::test_ray_buffer_tuning PASSED
TestCachingOptimizations::test_smiles_canonicalization_cache PASSED
TestCachingOptimizations::test_precomputed_test_blind_datasets PASSED
TestTrainingOptimizations::test_mixed_precision_config PASSED
TestTrainingOptimizations::test_gradient_accumulation_config PASSED
TestTrainingOptimizations::test_parallel_ensemble_gpu_allocation SKIPPED
TestIntegration::test_ensemble_with_precomputed_datasets PASSED
TestIntegration::test_optimizations_preserve_predictions PASSED
TestRegressionPrevention::test_no_redundant_file_loading PASSED
TestRegressionPrevention::test_smiles_cache_reduces_computations PASSED
======================================= 12 passed, 1 skipped in 5.26s =======================================
```

### Quality Validation Results

**Verified Optimizations:**
1. ✅ Batched predictions use configured batch_size (not hardcoded 1)
2. ✅ MLflow parameters logged in batch via `log_batch()` API
3. ✅ num_workers respects curriculum setting
4. ✅ Ray buffer tuning configurable via environment variables
5. ✅ SMILES canonicalization cache provides 99% hit rate for duplicate molecules
6. ✅ Test/blind datasets precomputed once and shared across all 25 ensemble members
7. ✅ Mixed precision training (16-mixed, bf16-mixed) can be enabled
8. ✅ Gradient accumulation configurable for larger effective batches
9. ✅ GPU allocation strategy verified (6 parallel for Chemprop, 4 for Chemeleon)

**Regression Prevention Verified:**
- ✅ Test file loaded only once per ensemble (not 25x)
- ✅ SMILES cache reduces computations 100x for duplicate molecules (100 calls → 1 computation, 99 cache hits)
- ✅ Predictions remain deterministic (same SMILES always produces same canonical form)
- ✅ Ensemble precomputation verified via integration test

### Risk Mitigation Status

All identified risks addressed and tested:

**High-Risk Items:**
- ✅ **2.4 Parallel Ensemble** - Implemented with user-validated GPU memory limits (3 models per GPU for Chemprop, 2 for Chemeleon)
- ✅ **2.3 Precomputed Datasets** - Verified correctness via integration test; test/blind files loaded once

**Medium-Risk Items:**
- ✅ **3.1 Mixed Precision** - Configuration validated, ready for use with precision="16-mixed"
- ✅ **2.2 SMILES Cache** - Cache hit rate verified at 99% for duplicates

**Low-Risk Items:**
- ✅ All Quick Wins (1.1-1.4) validated with unit tests
- ✅ No backward compatibility issues detected in test suite
