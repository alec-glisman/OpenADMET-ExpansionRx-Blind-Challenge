# Performance Optimization Implementation Plan
## Additional Optimizations Phase

**Created:** 2026-01-06
**Goal:** Implement torch.compile, persistent workers, prediction caching, and Bayesian HPO warmstart capabilities

---

## Overview

This plan implements four additional performance optimizations on top of the already-completed Phase 1-3 optimizations:

1. **torch.compile Integration** - 20-40% additional speedup for both Chemprop and Chemeleon
2. **Persistent Workers** - Already implemented, verify and document
3. **Prediction Caching** - Reduce ensemble I/O overhead while preserving all outputs
4. **Bayesian HPO Warmstart** - Enable study persistence and continuation

**Expected Combined Impact:** 30-60% additional speedup on top of existing 2-3x gains

---

## Task 1: torch.compile Integration for Chemprop & Chemeleon

### Objective
Enable PyTorch 2.0+ model compilation for both Chemprop and Chemeleon models to achieve 20-40% training speedup through kernel fusion and reduced overhead.

### Implementation Details

#### 1.1 Add Configuration Schema

**File:** `src/admet/model/chemprop/config.py`
**Location:** Lines 291-336 (PerformanceOptimizationConfig dataclass)

**Changes:**
```python
@dataclass
class PerformanceOptimizationConfig:
    """Configuration for performance optimizations during training."""

    use_mixed_precision: bool = False
    async_checkpoint_upload: bool = False
    checkpoint_save_interval_seconds: float = 0.0

    # NEW: torch.compile settings
    use_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = False
```

**Validation:**
- `torch_compile_mode` must be one of: `"default"`, `"reduce-overhead"`, `"max-autotune"`
- Add `__post_init__` validator if needed

**Notes:**
- `reduce-overhead`: Recommended for training (1.5-2.0x speedup, minimal warmup)
- `max-autotune`: Maximum speedup but longer compilation (1.8-2.5x)
- `default`: Conservative baseline (1.3-1.8x)

#### 1.2 Integrate torch.compile in Chemprop

**File:** `src/admet/model/chemprop/model.py`
**Location:** Lines 1322-1374 (_prepare_model method)

**Changes:**
Add compilation immediately after MPNN instantiation (after line 1374):

```python
def _prepare_model(self) -> None:
    """Build the MPNN model based on hyperparameters."""

    # ... existing code for FFN, task affinity, etc. ...

    # Line 1363-1374: Create MPNN
    self.mpnn = MPNNWithWeightDecay(
        message_passing=self.mp,
        agg=self.agg,
        predictor=self.ffn,
        batch_norm=self.hyperparams.batch_norm,
        metrics=self.metrics,
        warmup_epochs=self.hyperparams.warmup_epochs,
        init_lr=self.hyperparams.init_lr,
        max_lr=self.hyperparams.max_lr,
        final_lr=self.hyperparams.final_lr,
        weight_decay=self.hyperparams.weight_decay,
    )

    # NEW: Apply torch.compile if enabled
    if self._performance_optimization.use_torch_compile:
        import torch
        logger.info(
            "Compiling MPNN with torch.compile (mode=%s, fullgraph=%s)",
            self._performance_optimization.torch_compile_mode,
            self._performance_optimization.torch_compile_fullgraph,
        )
        self.mpnn = torch.compile(
            self.mpnn,
            mode=self._performance_optimization.torch_compile_mode,
            fullgraph=self._performance_optimization.torch_compile_fullgraph,
            dynamic=self._performance_optimization.torch_compile_dynamic,
        )
        logger.info("MPNN compilation complete - expect 20-40%% training speedup")
```

**Error Handling:**
- Wrap in try/except to handle compilation failures gracefully
- Fall back to uncompiled model with warning if compilation fails
- Log compilation time for performance tracking

#### 1.3 Integrate torch.compile in Chemeleon

**File:** `src/admet/model/chemeleon/model.py`
**Location:** Lines 471-544 (_init_model method)

**Changes:**
Add compilation after MPNN creation (after line 544):

```python
def _init_model(self, n_tasks: int) -> None:
    """Initialize model components."""

    # ... existing code for loading pretrained MP, FFN, etc. ...

    # Line 534-544: Create MPNN
    self.mpnn = models.MPNN(
        message_passing=self.mp,
        agg=self.agg,
        predictor=self.ffn,
        batch_norm=self._get_model_param("batch_norm", False),
        metrics=metrics,
        warmup_epochs=warmup_epochs,
        init_lr=init_lr,
        max_lr=max_lr,
        final_lr=final_lr,
    )

    # NEW: Apply torch.compile if enabled
    perf_config = self.config.model.chemeleon.get("performance_optimization", {})
    use_compile = perf_config.get("use_torch_compile", False)

    if use_compile:
        import torch
        compile_mode = perf_config.get("torch_compile_mode", "reduce-overhead")
        fullgraph = perf_config.get("torch_compile_fullgraph", False)
        dynamic = perf_config.get("torch_compile_dynamic", False)

        logger.info(
            "Compiling Chemeleon MPNN with torch.compile (mode=%s)",
            compile_mode,
        )
        self.mpnn = torch.compile(
            self.mpnn,
            mode=compile_mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        logger.info("Chemeleon MPNN compilation complete")
```

**Note:** Chemeleon uses nested config access, adapt to existing config structure.

#### 1.4 Update Configuration Files

**Files to Update:**
- `configs/2-hpo-ensemble/ensemble_chemprop_hpo_*.yaml` (all ensemble configs)
- `configs/3-production/ensemble_chemprop_*.yaml`
- `configs/0-experiment/chemprop.yaml`
- `configs/0-experiment/chemeleon.yaml` (if exists)

**Add Section:**
```yaml
performance_optimization:
  use_mixed_precision: true          # Already enabled
  use_torch_compile: true            # NEW
  torch_compile_mode: "reduce-overhead"  # Recommended for training
  torch_compile_fullgraph: false     # Set true for small models only
  torch_compile_dynamic: false       # Set true for variable input sizes
```

**Priority Configs to Update:**
1. Production ensemble configs (highest impact)
2. HPO ensemble configs (validate before mass HPO runs)
3. Experimental configs (for testing)

#### 1.5 Testing & Validation

**Test Plan:**
1. **Compilation Test:**
   - Train single model with `use_torch_compile: true`
   - Verify no errors during compilation
   - Check logs for "MPNN compilation complete" message

2. **Performance Benchmark:**
   - Train same config with compile=false vs compile=true
   - Measure wall-clock time per epoch
   - Target: 20-40% speedup (combined with FP16: 1.8-2.5x total)

3. **Prediction Equivalence:**
   - Train two models with same seed: one compiled, one not
   - Compare predictions on test set
   - Assert predictions match within 1e-4 tolerance

4. **Ensemble Compatibility:**
   - Run 2-model mini-ensemble with compilation enabled
   - Verify Ray parallelization works correctly
   - Check all plots and outputs generated

**Validation Script:**
```bash
# Test Chemprop compilation
admet model train -c configs/0-experiment/chemprop_compile_test.yaml

# Test Chemeleon compilation
admet model train -c configs/0-experiment/chemeleon_compile_test.yaml

# Mini ensemble test
admet model ensemble -c configs/validation/mini_ensemble_compile.yaml
```

**Success Criteria:**
- ✅ Models compile without errors
- ✅ Training completes successfully
- ✅ 20-40% speedup observed
- ✅ Predictions match uncompiled version
- ✅ All ensemble outputs preserved

---

## Task 2: Persistent Workers Verification

### Objective
Verify that persistent_workers is already correctly implemented and document usage.

### Current Implementation Status

**Already Implemented:** ✅

**Files:**
- `src/admet/model/chemprop/model.py` (lines 308-312)
- `src/admet/model/chemeleon/model.py` (lines 91-95)

**Implementation:**
```python
# Enable persistent_workers and prefetch_factor when using multiprocessing
if num_workers > 0:
    kwargs["persistent_workers"] = True
    kwargs["prefetch_factor"] = 2 if is_train else 1
```

### Verification Tasks

1. **Code Review:**
   - Confirm persistent_workers is enabled in both Chemprop and Chemeleon
   - Verify conditional logic (only when num_workers > 0)
   - Check prefetch_factor settings

2. **Documentation Update:**
   - Add note to performance optimization docs
   - Explain why num_workers > 0 is required
   - Document expected 5-10% speedup benefit

3. **No Code Changes Required**
   - This optimization is already correctly implemented
   - Only documentation updates needed

---

## Task 3: Ensemble Prediction Caching

### Objective
Add in-memory prediction caching to reduce I/O overhead during ensemble aggregation while preserving all file outputs and plots.

### Implementation Details

#### 3.1 Add Cache Attributes

**File:** `src/admet/model/chemprop/ensemble.py`
**Location:** Lines 200-300 (__init__ method)

**Changes:**
```python
class ModelEnsemble:
    def __init__(self, config: UnifiedModelConfig):
        # ... existing initialization ...

        # NEW: Prediction cache for in-memory access
        self._prediction_cache: Dict[str, Dict[str, pd.DataFrame]] = {
            "test": {},   # model_key -> predictions DataFrame
            "blind": {}   # model_key -> predictions DataFrame
        }
        self._aggregated_cache: Dict[str, pd.DataFrame] = {
            "test": None,
            "blind": None
        }
        self._cache_enabled: bool = True  # Can be disabled for debugging
```

#### 3.2 Cache Individual Predictions

**File:** `src/admet/model/chemprop/ensemble.py`
**Location:** Lines 1186-1273 (train_all method, result collection)

**Changes:**
Add caching after collecting results from Ray workers:

```python
def train_all(self) -> None:
    """Train all ensemble members and collect results."""

    # ... existing code for launching Ray tasks ...

    # Lines 1256-1261: Collect results
    for result in ready_results:
        model_key, metrics, test_preds, blind_preds, profiling, mlflow_id = result

        # Store metrics and MLflow info (existing code)
        self._all_metrics[model_key] = metrics
        self._mlflow_run_ids[model_key] = mlflow_id

        # Store predictions in lists (existing code)
        if test_preds is not None:
            self._all_test_predictions.append(test_preds)
        if blind_preds is not None:
            self._all_blind_predictions.append(blind_preds)

        # NEW: Cache individual predictions for fast lookup
        if self._cache_enabled:
            if test_preds is not None:
                self._prediction_cache["test"][model_key] = test_preds
            if blind_preds is not None:
                self._prediction_cache["blind"][model_key] = blind_preds

        # ... rest of existing code ...
```

#### 3.3 Cache Aggregated Predictions

**File:** `src/admet/model/chemprop/ensemble.py`
**Location:** Lines 1361-1432 (_aggregate_predictions method)

**Changes:**
Add cache check and storage:

```python
def _aggregate_predictions(
    self,
    predictions_list: list[pd.DataFrame],
    split_name: str,
    target_cols: list[str],
) -> pd.DataFrame:
    """Aggregate predictions from multiple models."""

    # NEW: Check cache first
    if self._cache_enabled and self._aggregated_cache[split_name] is not None:
        logger.debug("Using cached aggregated predictions for %s", split_name)
        return self._aggregated_cache[split_name]

    # ... existing aggregation logic ...

    # Create final DataFrame with all columns
    ensemble_df = pd.DataFrame({
        "SMILES": smiles,
        **{f"{col}_mean": means[col] for col in target_cols},
        **{f"{col}_std": stds[col] for col in target_cols},
        **{f"{col}_stderr": stderrs[col] for col in target_cols},
        **{f"{col}_transformed_mean": transformed_means.get(col) for col in target_cols if col in transformed_means},
        **{f"{col}_transformed_stderr": transformed_stderrs.get(col) for col in target_cols if col in transformed_stderrs},
    })

    # NEW: Cache aggregated result
    if self._cache_enabled:
        self._aggregated_cache[split_name] = ensemble_df
        logger.debug("Cached aggregated predictions for %s", split_name)

    return ensemble_df
```

#### 3.4 Add Cache Access Methods

**File:** `src/admet/model/chemprop/ensemble.py`
**Location:** Add new methods after _aggregate_predictions

**New Methods:**
```python
def get_cached_predictions(
    self,
    model_key: Optional[str] = None,
    split_name: str = "test",
) -> Optional[pd.DataFrame]:
    """
    Get cached predictions for a specific model or aggregated ensemble.

    Parameters
    ----------
    model_key : str, optional
        Model key (e.g., "split_0_fold_0"). If None, returns aggregated predictions.
    split_name : str
        Either "test" or "blind"

    Returns
    -------
    pd.DataFrame or None
        Cached predictions, or None if not in cache
    """
    if model_key is None:
        return self._aggregated_cache.get(split_name)
    else:
        return self._prediction_cache[split_name].get(model_key)

def clear_cache(self, split_name: Optional[str] = None) -> None:
    """
    Clear prediction cache.

    Parameters
    ----------
    split_name : str, optional
        If specified, clear only this split. Otherwise clear all.
    """
    if split_name is None:
        self._prediction_cache = {"test": {}, "blind": {}}
        self._aggregated_cache = {"test": None, "blind": None}
        logger.info("Cleared all prediction caches")
    else:
        self._prediction_cache[split_name] = {}
        self._aggregated_cache[split_name] = None
        logger.info("Cleared %s prediction cache", split_name)
```

#### 3.5 Preserve All File Outputs

**Critical Requirement:** All existing file outputs MUST be preserved.

**Files to Keep:**
1. ✅ Individual model predictions: `{split_name}_predictions.csv` (25 files per ensemble)
2. ✅ Individual model plots: Parity plots logged to MLflow (per model)
3. ✅ Ensemble predictions: `{split_name}_ensemble_predictions.csv`
4. ✅ Submission CSVs: `{split_name}_ensemble_submissions.csv`
5. ✅ Ensemble plots:
   - Unlabeled: `prediction_distributions.png`, `uncertainty_distributions.png`
   - Labeled: `parity_{target}.png`, `ensemble_{metric}.png`

**Verification:**
- No changes to `_save_ensemble_predictions()` method (lines 1434-1470)
- No changes to `_generate_unlabeled_ensemble_plots()` (lines 1472-1537)
- No changes to `_generate_ensemble_plots()` (lines 1612-1714)
- Cache is purely additive (in-memory only)

#### 3.6 Testing

**Test Cases:**
1. **Cache Population:**
   - Run ensemble training
   - Verify cache contains 25 test predictions
   - Verify cache contains aggregated results

2. **Cache Retrieval:**
   - Access predictions via `get_cached_predictions()`
   - Verify returns correct DataFrames
   - Check performance improvement for repeated access

3. **File Outputs Preserved:**
   - Run ensemble with caching enabled
   - Verify all 25 individual CSVs written
   - Verify ensemble CSVs written
   - Verify all plots generated and logged to MLflow

4. **Cache Clearing:**
   - Clear cache and verify empty
   - Re-run aggregation and verify cache repopulated

**Success Criteria:**
- ✅ Predictions cached in memory
- ✅ All file outputs identical to pre-caching implementation
- ✅ All plots generated correctly
- ✅ MLflow artifacts logged as before
- ✅ Faster repeated access to predictions (if needed)

---

## Task 4: Bayesian HPO Warmstart with Persistent Studies

### Objective
Enable Optuna study persistence to SQLite database, add warmstart capability to continue previous HPO runs, create CLI commands for study management, and document usage.

### Implementation Details

#### 4.1 Extend SearchAlgorithmConfig

**File:** `src/admet/model/chemprop/hpo_config.py`
**Location:** Lines 148-164 (SearchAlgorithmConfig dataclass)

**Changes:**
```python
@dataclass
class SearchAlgorithmConfig:
    """
    Configuration for Ray Tune search algorithm.

    Enables Bayesian optimization (Optuna) or other adaptive search methods
    instead of pure random sampling. This can significantly improve HPO
    efficiency by learning which hyperparameter regions perform well.

    NEW: Supports persistent Optuna studies for warmstarting optimization
    from previous runs.

    Attributes
    ----------
    type : str
        Search algorithm type - "random", "optuna", "bayesopt", "hyperopt"
    seed : int
        Random seed for reproducibility
    n_initial_points : int
        Number of random trials before using surrogate model (Optuna only)
    persist_study : bool
        Whether to save Optuna study to persistent storage (SQLite database)
    study_name : str, optional
        Name for the Optuna study. If None, auto-generated with timestamp.
    storage_dir : str, optional
        Directory for SQLite database. Defaults to {output_dir}/optuna_studies
    warmstart_from : str, optional
        Name of previous study to warmstart from (loads top trials as seeds)
    warmstart_n_trials : int
        Number of top trials to enqueue from previous study for warmstart
    """

    type: str = "optuna"
    seed: int = 42
    n_initial_points: int = 20

    # NEW: Study persistence settings
    persist_study: bool = False
    study_name: Optional[str] = None
    storage_dir: Optional[str] = None
    warmstart_from: Optional[str] = None
    warmstart_n_trials: int = 10
```

**Apply same changes to:** `src/admet/model/chemeleon/hpo_config.py` (lines 153-169)

#### 4.2 Modify _build_search_algorithm for Persistence

**File:** `src/admet/model/chemprop/hpo.py`
**Location:** Lines 343-398 (_build_search_algorithm method)

**Changes:**
Replace existing Optuna initialization with persistent study support:

```python
def _build_search_algorithm(self) -> SearchAlgorithm | None:
    """Build the Ray Tune search algorithm."""

    algo_type = getattr(self.config.search_algorithm, "type", "random")

    if algo_type == "random" or algo_type == "none":
        logger.info("Random search algorithm selected")
        return None

    elif algo_type == "optuna":
        import optuna
        from ray.tune.search.optuna import OptunaSearch
        from datetime import datetime

        # Determine storage location
        if self.config.search_algorithm.persist_study:
            storage_dir = Path(self.config.search_algorithm.storage_dir or self.config.output_dir) / "optuna_studies"
            storage_dir.mkdir(parents=True, exist_ok=True)
            storage_url = f"sqlite:///{storage_dir / 'studies.db'}"

            # Generate or use provided study name
            study_name = self.config.search_algorithm.study_name
            if study_name is None:
                study_name = f"hpo_{datetime.now():%Y%m%d_%H%M%S}"

            logger.info(
                "Creating persistent Optuna study: %s (storage: %s)",
                study_name,
                storage_url,
            )
        else:
            storage_url = None
            study_name = None
            logger.info("Using ephemeral Optuna study (no persistence)")

        # Create sampler
        sampler = optuna.samplers.TPESampler(
            seed=self.config.search_algorithm.seed,
            n_startup_trials=self.config.search_algorithm.n_initial_points,
        )

        # Determine direction
        direction = "minimize" if self.config.asha.mode == "min" else "maximize"

        # Create or load study
        if storage_url and study_name:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                sampler=sampler,
                direction=direction,
                load_if_exists=False,  # Error if study exists (prevents accidental overwrite)
            )

            # Warmstart from previous study if specified
            warmstart_from = self.config.search_algorithm.warmstart_from
            if warmstart_from:
                logger.info("Warmstarting from study: %s", warmstart_from)
                try:
                    old_study = optuna.load_study(
                        study_name=warmstart_from,
                        storage=storage_url,
                    )

                    # Get top N trials from previous study
                    n_warmstart = self.config.search_algorithm.warmstart_n_trials
                    top_trials = old_study.best_trials[:n_warmstart]

                    logger.info(
                        "Enqueuing %d top trials from %s (best value: %.4f)",
                        len(top_trials),
                        warmstart_from,
                        old_study.best_value,
                    )

                    # Enqueue trials as seeds
                    for trial in top_trials:
                        study.enqueue_trial(trial.params)

                except Exception as e:
                    logger.warning(
                        "Failed to load warmstart study %s: %s. Continuing without warmstart.",
                        warmstart_from,
                        e,
                    )

            # Create OptunaSearch with persistent study
            search_alg = OptunaSearch(
                study=study,
                metric=self.config.asha.metric,
                mode=self.config.asha.mode,
            )
        else:
            # Ephemeral study (original behavior)
            search_alg = OptunaSearch(
                sampler=sampler,
                metric=self.config.asha.metric,
                mode=self.config.asha.mode,
            )

        logger.info(
            "Using Optuna search algorithm (TPESampler) with %d initial random points",
            self.config.search_algorithm.n_initial_points,
        )
        return search_alg

    # ... rest of existing code for bayesopt, hyperopt ...
```

**Apply equivalent changes to:** `src/admet/model/chemeleon/hpo.py` (lines 646-698)

#### 4.3 Log Study Metadata

**File:** `src/admet/model/chemprop/hpo.py`
**Location:** Lines 442-536 (_log_results method)

**Changes:**
Add study metadata logging after results are saved:

```python
def _log_results(self) -> None:
    """Log HPO results to MLflow and save to disk."""

    # ... existing code for saving hpo_results.csv, top_k_configs.json ...

    # NEW: Log study metadata if persistence enabled
    if self.config.search_algorithm.persist_study:
        study_metadata = {
            "study_name": self.config.search_algorithm.study_name,
            "storage_dir": str(Path(self.config.search_algorithm.storage_dir or self.config.output_dir) / "optuna_studies"),
            "n_trials": len(self.results),
            "best_metric": self.results.get_best_result().metrics[self.config.asha.metric],
            "warmstart_from": self.config.search_algorithm.warmstart_from,
        }

        # Save to JSON
        metadata_path = Path(self.config.output_dir) / "study_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(study_metadata, f, indent=2)

        logger.info("Study metadata saved to %s", metadata_path)
        logger.info(
            "To warmstart from this study, use: warmstart_from: '%s'",
            self.config.search_algorithm.study_name,
        )
```

#### 4.4 Add CLI Command for Study Management

**File:** `src/admet/cli/model.py`
**Location:** Add new command after hpo command (after line 378)

**New Command:**
```python
@model_app.command(name="hpo-list-studies")
def hpo_list_studies(
    storage_dir: Path = typer.Option(
        None,
        "--storage-dir",
        "-s",
        help="Directory containing optuna_studies/studies.db (defaults to current dir)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed trial info"),
) -> None:
    """
    List all available Optuna studies in the database.

    Shows study names, number of trials, best metrics, and timestamps
    to help identify which study to use for warmstarting.

    Examples:
        # List all studies in default location
        admet model hpo-list-studies

        # List studies in specific directory
        admet model hpo-list-studies --storage-dir hpo_results/

        # Show detailed trial information
        admet model hpo-list-studies --verbose
    """
    import optuna
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Determine storage path
    if storage_dir is None:
        storage_dir = Path.cwd() / "optuna_studies"
    else:
        storage_dir = Path(storage_dir) / "optuna_studies"

    db_path = storage_dir / "studies.db"

    if not db_path.exists():
        console.print(f"[red]No Optuna database found at {db_path}[/red]")
        console.print(f"[yellow]Hint: Run HPO with persist_study: true to create studies[/yellow]")
        raise typer.Exit(1)

    storage_url = f"sqlite:///{db_path}"

    try:
        summaries = optuna.get_all_study_summaries(storage=storage_url)
    except Exception as e:
        console.print(f"[red]Failed to load studies: {e}[/red]")
        raise typer.Exit(1)

    if not summaries:
        console.print("[yellow]No studies found in database[/yellow]")
        raise typer.Exit(0)

    # Create table
    table = Table(title=f"Optuna Studies ({len(summaries)} found)")
    table.add_column("Study Name", style="cyan", no_wrap=True)
    table.add_column("N Trials", justify="right", style="magenta")
    table.add_column("Best Value", justify="right", style="green")
    table.add_column("Direction", justify="center")
    table.add_column("Created", style="blue")

    for summary in summaries:
        best_value = f"{summary.best_trial.value:.4f}" if summary.best_trial else "N/A"
        created = summary.datetime_start.strftime("%Y-%m-%d %H:%M") if summary.datetime_start else "Unknown"

        table.add_row(
            summary.study_name,
            str(summary.n_trials),
            best_value,
            summary.direction.name,
            created,
        )

    console.print(table)

    if verbose:
        console.print("\n[bold]Study Details:[/bold]\n")
        for summary in summaries:
            console.print(f"[cyan]{summary.study_name}[/cyan]:")
            if summary.best_trial:
                console.print(f"  Best params: {summary.best_trial.params}")
            console.print(f"  System attrs: {summary.system_attrs}")
            console.print()

    console.print(f"\n[green]Database location:[/green] {db_path}")
    console.print(f"[yellow]To warmstart from a study, add to your config:[/yellow]")
    console.print("  search_algorithm:")
    console.print("    warmstart_from: '<study_name>'")
```

**Update CLI exports:**
Add the new command to `__all__` if needed.

#### 4.5 Create Documentation

**File:** `docs/guide/hpo_warmstart.rst`
**Create new file**

**Content:**
```rst
Warmstarting Hyperparameter Optimization
=========================================

Overview
--------

Warmstarting enables you to continue hyperparameter optimization from previous runs,
leveraging historical trial data to accelerate convergence. This is especially useful when:

- Refining hyperparameters after initial broad search
- Adding more trials to existing studies
- Exploring neighboring regions of known-good configurations

Key Benefits
^^^^^^^^^^^^

- **30-50% fewer trials** to reach optimal configurations
- Reuse expensive trial evaluations from previous runs
- Iteratively refine search without starting from scratch
- Build institutional knowledge of hyperparameter landscapes

How It Works
------------

Warmstarting uses Optuna's persistent study storage:

1. **Persistent Studies**: Studies are saved to SQLite database
2. **Trial History**: All trial parameters and results stored
3. **Top-K Seeding**: Best trials from previous study enqueued first
4. **Bayesian Continuation**: TPE sampler uses historical data for suggestions

Configuration
-------------

Basic Setup
^^^^^^^^^^^

Enable study persistence in your HPO config:

.. code-block:: yaml

   search_algorithm:
     type: optuna
     seed: 42
     n_initial_points: 20

     # NEW: Persistence settings
     persist_study: true                    # Enable study saving
     study_name: "chemprop_v1_initial"      # Unique study identifier
     storage_dir: "hpo_results"             # Database location

This creates: ``hpo_results/optuna_studies/studies.db``

Warmstart from Previous Study
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To continue from a previous study:

.. code-block:: yaml

   search_algorithm:
     type: optuna
     persist_study: true
     study_name: "chemprop_v1_continued"    # NEW study name (required)
     storage_dir: "hpo_results"             # SAME location as original
     warmstart_from: "chemprop_v1_initial"  # Previous study to load
     warmstart_n_trials: 10                 # Top 10 trials enqueued first

CLI Usage
---------

List Available Studies
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # List all studies in default location
   admet model hpo-list-studies

   # List studies in specific directory
   admet model hpo-list-studies --storage-dir hpo_results/

   # Show detailed trial information
   admet model hpo-list-studies --verbose

Example output::

   ┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
   ┃ Study Name             ┃ N Trials┃ Best Value ┃ Direction ┃ Created         ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
   │ chemprop_v1_initial    │      50 │     0.4523 │ MINIMIZE  │ 2026-01-05 14:30│
   │ chemprop_v1_continued  │     100 │     0.4389 │ MINIMIZE  │ 2026-01-06 09:15│
   └────────────────────────┴─────────┴────────────┴───────────┴─────────────────┘

Run HPO with Warmstart
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run initial study
   admet model hpo -c configs/hpo_initial.yaml --num-samples 50

   # Note the study name from study_metadata.json or CLI output

   # Continue with warmstart
   admet model hpo -c configs/hpo_warmstart.yaml --num-samples 50

Workflow Examples
-----------------

Iterative Refinement
^^^^^^^^^^^^^^^^^^^^

**Phase 1: Broad Search**

.. code-block:: yaml

   # configs/hpo_phase1.yaml
   search_algorithm:
     persist_study: true
     study_name: "chemprop_broad_search"

   search_space:
     learning_rate:
       type: loguniform
       low: 0.0001
       high: 0.1
     hidden_dim:
       type: choice
       categories: [128, 256, 512, 1024]

   resources:
     num_samples: 100

**Phase 2: Focused Search**

.. code-block:: yaml

   # configs/hpo_phase2.yaml
   search_algorithm:
     persist_study: true
     study_name: "chemprop_focused_search"
     warmstart_from: "chemprop_broad_search"
     warmstart_n_trials: 15  # Top 15 as seeds

   search_space:
     learning_rate:
       type: loguniform
       low: 0.001   # Narrowed based on phase 1
       high: 0.01
     hidden_dim:
       type: choice
       categories: [256, 512]  # Top performers only

   resources:
     num_samples: 50  # Fewer trials needed

Cross-Validation Study
^^^^^^^^^^^^^^^^^^^^^^

Warmstart across different data splits:

.. code-block:: yaml

   # configs/hpo_fold0.yaml
   search_algorithm:
     persist_study: true
     study_name: "chemprop_fold0"

   data:
     data_dir: "data/split_0/fold_0"

.. code-block:: yaml

   # configs/hpo_fold1.yaml
   search_algorithm:
     persist_study: true
     study_name: "chemprop_fold1"
     warmstart_from: "chemprop_fold0"  # Use fold0 results

   data:
     data_dir: "data/split_0/fold_1"

Best Practices
--------------

Study Naming
^^^^^^^^^^^^

- Use descriptive names: ``{model}_{dataset}_{version}``
- Include timestamps for experiments: ``chemprop_v1_20260106``
- Document rationale in config comments

Storage Management
^^^^^^^^^^^^^^^^^^

- Keep one ``studies.db`` per project
- Backup before major changes
- Archive completed studies to separate database

Warmstart Tuning
^^^^^^^^^^^^^^^^

- Start with 10-20 warmstart trials (``warmstart_n_trials``)
- More trials = faster initial convergence but less exploration
- Fewer trials = more exploration but slower start

Monitoring
^^^^^^^^^^

- Compare warmstart vs cold-start performance
- Track trials-to-convergence metric
- Log study metadata for reproducibility

Integration with MLflow
-----------------------

Study metadata is automatically logged to MLflow:

.. code-block:: python

   # Logged as MLflow params
   mlflow.log_param("optuna_study_name", "chemprop_v1_initial")
   mlflow.log_param("optuna_storage_dir", "hpo_results/optuna_studies")
   mlflow.log_param("warmstart_from", "chemprop_v1_initial")

Use MLflow UI to track:

- Study lineage (which studies warmstarted from others)
- Convergence comparisons
- Best trial evolution

Troubleshooting
---------------

Study Not Found
^^^^^^^^^^^^^^^

**Error**: ``Study 'chemprop_v1' not found in database``

**Solution**: Verify study name with ``admet model hpo-list-studies``

Study Already Exists
^^^^^^^^^^^^^^^^^^^^

**Error**: ``Study 'chemprop_v2' already exists``

**Solution**: Use different study name or delete existing study:

.. code-block:: python

   import optuna
   optuna.delete_study(study_name="chemprop_v2", storage="sqlite:///...")

Incompatible Search Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Warning**: ``Parameter 'hidden_dim' not in warmstart trials``

**Solution**: Ensure search spaces are compatible. Warmstart trials will be skipped
if parameters don't match new search space.

API Reference
-------------

See :class:`admet.model.chemprop.hpo_config.SearchAlgorithmConfig` for detailed
configuration options.

Related Documentation
---------------------

- :doc:`hpo` - Main HPO guide
- :doc:`configuration` - Configuration file structure
- `Optuna Documentation <https://optuna.readthedocs.io/>`_ - Underlying framework
```

#### 4.6 Update Main HPO Documentation

**File:** `docs/guide/hpo.rst`
**Location:** Add new section after line 476 (Best Practices)

**Add:**
```rst
Warmstarting Optimization
-------------------------

Continue optimization from previous runs using persistent Optuna studies.
See :doc:`hpo_warmstart` for detailed guide.

Quick example:

.. code-block:: yaml

   search_algorithm:
     persist_study: true
     study_name: "chemprop_v2"
     warmstart_from: "chemprop_v1"  # Load top trials from v1
     warmstart_n_trials: 10

Benefits:

- 30-50% fewer trials to reach optimal configuration
- Iteratively refine search without starting from scratch
- Build on previous experiments

Commands:

.. code-block:: bash

   # List available studies
   admet model hpo-list-studies

   # Run HPO with warmstart
   admet model hpo -c configs/hpo_warmstart.yaml
```

#### 4.7 Example Configuration Files

**Create:** `configs/1-hpo-single/hpo_chemprop_warmstart_example.yaml`

```yaml
# Example: Warmstart HPO from previous study
experiment_name: chemprop_hpo_warmstart_example
output_dir: hpo_results/warmstart_example
mlflow_tracking_uri: http://127.0.0.1:8084

data:
  data_file: data/training_data.csv
  smiles_col: SMILES
  target_cols:
    - LogD
    - Log KSOL
    - Log HLM CLint
  # ... other data config ...

model:
  type: chemprop
  chemprop:
    message_hidden_dim: 300
    ffn_type: regression
    # ... other model config ...

search_space:
  # Narrowed search space based on previous study results
  learning_rate:
    type: loguniform
    low: 0.001
    high: 0.01

  hidden_dim:
    type: choice
    categories: [256, 512]

  depth:
    type: randint
    low: 3
    high: 6

asha:
  metric: val_mae
  mode: min
  max_t: 120
  grace_period: 15
  reduction_factor: 3

search_algorithm:
  type: optuna
  seed: 42
  n_initial_points: 10  # Reduced since warmstart provides good starting points

  # Persistence & warmstart settings
  persist_study: true
  study_name: "chemprop_focused_v2"
  storage_dir: "hpo_results"
  warmstart_from: "chemprop_broad_v1"  # Previous study name
  warmstart_n_trials: 15  # Enqueue top 15 trials as seeds

resources:
  num_samples: 50  # Fewer trials needed with warmstart
  max_concurrent_trials: 4
  cpus_per_trial: 4
  gpus_per_trial: 0.33

transfer_learning:
  enabled: false  # Will enable after finding optimal config
```

#### 4.8 Testing

**Test Plan:**

1. **Study Persistence Test:**
   ```bash
   # Run HPO with persistence
   admet model hpo -c configs/test_persistence.yaml --num-samples 10

   # Verify database created
   ls hpo_results/optuna_studies/studies.db

   # List studies
   admet model hpo-list-studies --storage-dir hpo_results/
   ```

2. **Warmstart Test:**
   ```bash
   # Run initial study
   admet model hpo -c configs/test_initial.yaml --num-samples 20

   # Run warmstart study (should enqueue top trials first)
   admet model hpo -c configs/test_warmstart.yaml --num-samples 10

   # Verify warmstart worked (check logs for "Enqueuing X trials")
   ```

3. **CLI Test:**
   ```bash
   # List studies with verbose output
   admet model hpo-list-studies --storage-dir hpo_results/ --verbose

   # Verify output shows study details
   ```

4. **Cross-Study Compatibility:**
   - Run study with search_space A
   - Warmstart with overlapping search_space B
   - Verify compatible parameters used, incompatible ones ignored

**Success Criteria:**
- ✅ Studies persist to SQLite database
- ✅ CLI lists studies correctly
- ✅ Warmstart enqueues top trials
- ✅ HPO runs complete successfully
- ✅ Metadata logged to study_metadata.json
- ✅ Documentation renders correctly

---

## Implementation Timeline

### Phase 1: torch.compile (Day 1-2)
- **Day 1 Morning:** Add config schema, implement Chemprop integration
- **Day 1 Afternoon:** Implement Chemeleon integration, update configs
- **Day 2 Morning:** Testing and validation
- **Day 2 Afternoon:** Performance benchmarking

### Phase 2: Prediction Caching (Day 2-3)
- **Day 2 Afternoon:** Add cache attributes and methods
- **Day 3 Morning:** Integrate caching in ensemble workflow
- **Day 3 Afternoon:** Testing and validation

### Phase 3: HPO Warmstart (Day 3-5)
- **Day 3 Afternoon:** Extend config schema
- **Day 4 Morning:** Modify _build_search_algorithm
- **Day 4 Afternoon:** Add CLI command and study management
- **Day 5 Morning:** Create documentation
- **Day 5 Afternoon:** Testing and example configs

### Total Estimated Time: 4-5 days

---

## Testing & Validation

### Comprehensive Test Suite

**Create:** `tests/model/test_additional_optimizations.py`

```python
"""Tests for additional performance optimizations: torch.compile, caching, warmstart."""

import pytest
import torch
import pandas as pd
from pathlib import Path

class TestTorchCompile:
    """Test torch.compile integration."""

    def test_chemprop_compilation_enabled(self, sample_config):
        """Test that Chemprop model compiles when enabled."""
        config = sample_config
        config.model.chemprop.performance_optimization.use_torch_compile = True
        config.model.chemprop.performance_optimization.torch_compile_mode = "default"

        from admet.model.chemprop.model import ChempropModel
        model = ChempropModel.from_config(config)

        # Verify MPNN is compiled
        assert hasattr(model.mpnn, "_compiled")

    def test_compilation_modes(self):
        """Test all torch.compile modes are valid."""
        valid_modes = ["default", "reduce-overhead", "max-autotune"]
        for mode in valid_modes:
            # Should not raise
            torch.compile(torch.nn.Linear(10, 10), mode=mode)

    def test_prediction_equivalence(self, sample_data):
        """Test compiled and uncompiled models produce same predictions."""
        # Train both versions with same seed
        # Compare predictions
        pass

class TestPredictionCaching:
    """Test ensemble prediction caching."""

    def test_cache_initialization(self, ensemble_config):
        """Test cache attributes are initialized."""
        from admet.model.chemprop.ensemble import ModelEnsemble
        ensemble = ModelEnsemble(ensemble_config)

        assert hasattr(ensemble, "_prediction_cache")
        assert "test" in ensemble._prediction_cache
        assert "blind" in ensemble._prediction_cache
        assert hasattr(ensemble, "_aggregated_cache")

    def test_cache_population(self, ensemble_with_results):
        """Test cache is populated after training."""
        # Run ensemble training
        # Verify cache contains predictions
        pass

    def test_cache_retrieval(self, ensemble_with_cache):
        """Test cache retrieval methods."""
        # Get cached predictions
        test_preds = ensemble_with_cache.get_cached_predictions(split_name="test")
        assert test_preds is not None
        assert isinstance(test_preds, pd.DataFrame)

    def test_file_outputs_preserved(self, ensemble_output_dir):
        """Test all file outputs are still created."""
        # Verify CSVs, plots, MLflow artifacts all exist
        pass

class TestHPOWarmstart:
    """Test Bayesian HPO warmstart functionality."""

    def test_study_persistence(self, hpo_config_with_persistence):
        """Test Optuna study is saved to database."""
        # Run HPO
        # Verify studies.db exists
        # Verify study can be loaded
        pass

    def test_warmstart_enqueues_trials(self, hpo_config_with_warmstart):
        """Test warmstart enqueues top trials from previous study."""
        # Run initial study
        # Run warmstart study
        # Verify first N trials match top N from previous
        pass

    def test_cli_list_studies(self, temp_study_db):
        """Test CLI command lists studies correctly."""
        # Run CLI command
        # Parse output
        # Verify study names, metrics shown
        pass

class TestIntegration:
    """Integration tests for all optimizations together."""

    def test_all_optimizations_enabled(self, full_config):
        """Test all optimizations work together."""
        config = full_config
        config.model.chemprop.performance_optimization.use_mixed_precision = True
        config.model.chemprop.performance_optimization.use_torch_compile = True
        config.model.chemprop.optimization.num_workers = 4

        # Train mini ensemble
        # Verify all optimizations active
        # Verify outputs correct
        pass
```

---

## Success Criteria

### torch.compile
- ✅ Chemprop models compile successfully
- ✅ Chemeleon models compile successfully
- ✅ 20-40% speedup observed in benchmarks
- ✅ Predictions match uncompiled version (within tolerance)
- ✅ Works with mixed precision training
- ✅ All ensemble outputs preserved

### Prediction Caching
- ✅ Cache populated during ensemble training
- ✅ Cache retrieval methods work correctly
- ✅ All file outputs preserved (CSVs, plots)
- ✅ MLflow artifacts logged correctly
- ✅ No performance degradation

### HPO Warmstart
- ✅ Studies persist to SQLite database
- ✅ Warmstart loads and enqueues top trials
- ✅ CLI command lists studies correctly
- ✅ Documentation complete and accurate
- ✅ Example configs work end-to-end
- ✅ Metadata logged correctly

### Integration
- ✅ All optimizations work together
- ✅ No conflicts or errors
- ✅ End-to-end ensemble workflow succeeds
- ✅ HPO workflow succeeds with all features

---

## Files to Modify

| Priority | File | Changes |
|----------|------|---------|
| **P0** | `src/admet/model/chemprop/config.py` | Add torch_compile fields to PerformanceOptimizationConfig |
| **P0** | `src/admet/model/chemprop/model.py` | Add torch.compile() in _prepare_model() |
| **P0** | `src/admet/model/chemeleon/model.py` | Add torch.compile() in _init_model() |
| **P1** | `src/admet/model/chemprop/hpo_config.py` | Extend SearchAlgorithmConfig for persistence |
| **P1** | `src/admet/model/chemeleon/hpo_config.py` | Extend SearchAlgorithmConfig for persistence |
| **P1** | `src/admet/model/chemprop/hpo.py` | Modify _build_search_algorithm for warmstart |
| **P1** | `src/admet/model/chemeleon/hpo.py` | Modify _build_search_algorithm for warmstart |
| **P1** | `src/admet/model/chemprop/ensemble.py` | Add prediction caching |
| **P2** | `src/admet/cli/model.py` | Add hpo-list-studies command |
| **P2** | `docs/guide/hpo_warmstart.rst` | Create warmstart documentation |
| **P2** | `docs/guide/hpo.rst` | Add warmstart section |
| **P3** | `configs/**/*.yaml` | Update configs with new settings |
| **P3** | `tests/model/test_additional_optimizations.py` | Create test suite |

---

## Risk Mitigation

### torch.compile Risks
- **Compilation Failures:** Wrap in try/except, fall back to uncompiled
- **Incompatible Operations:** Use fullgraph=False to allow graph breaks
- **Numerical Differences:** Validate predictions match within tolerance

### Caching Risks
- **Memory Overhead:** Cache is in-memory only, cleared after ensemble completes
- **Stale Cache:** Clear cache methods provided for debugging

### Warmstart Risks
- **Database Corruption:** Use SQLite WAL mode for concurrent access
- **Incompatible Search Spaces:** Gracefully skip incompatible trials
- **Study Name Conflicts:** Require unique study names, error on duplicate

---

## Notes

### persistent_workers
- Already implemented correctly ✅
- No code changes needed
- Document in performance guide
- Verify behavior in tests

### torch.compile + Mixed Precision
- Synergistic effects: 1.8-2.5x combined speedup
- Both already enabled in production configs
- torch.compile adds minimal overhead on top of FP16

### Prediction Caching Philosophy
- **In-memory only** - no persistent cache to disk
- **Additive** - doesn't change existing behavior
- **Optional** - can be disabled via `_cache_enabled` flag
- **Read-only after population** - cache populated once during training

### HPO Warmstart Best Practices
- Use descriptive study names
- Keep one database per project
- Start with 10-20 warmstart trials
- Monitor convergence improvement
- Document study lineage

---

## Expected Performance Impact

| Optimization | Expected Speedup | Stacks With |
|--------------|------------------|-------------|
| torch.compile (alone) | 1.3-1.8x | - |
| torch.compile + FP16 | 1.8-2.5x | ✅ Synergistic |
| Prediction caching | 5-10% (I/O reduction) | ✅ All |
| HPO warmstart | 30-50% fewer trials | N/A |

**Combined Impact:** 30-60% additional speedup on already-optimized pipeline.

---

## Completion Checklist

**Completed (2026-01-06):**
- [x] torch.compile config schema added (Chemprop)
- [x] torch.compile integrated in Chemprop (_prepare_model)
- [x] torch.compile integrated in Chemeleon (_init_model)
- [x] Prediction cache attributes added (ModelEnsemble)
- [x] Prediction cache methods implemented (get_cached_predictions, clear_cache)
- [x] Cache integration in ensemble workflow (result collection + aggregation)
- [x] SearchAlgorithmConfig extended (Chemprop + Chemeleon)
- [x] _build_search_algorithm modified for persistence (Chemprop)

**In Progress:**
- [ ] _build_search_algorithm modified for persistence (Chemeleon)
- [ ] Study metadata logging added to _log_results
- [ ] CLI hpo-list-studies command added
- [ ] Warmstart documentation created
- [ ] Main HPO docs updated
- [ ] Example configs created
- [ ] Test suite created
- [ ] All tests passing
- [ ] Update all HPO YAML configs
- [ ] Performance benchmarks run
- [ ] Documentation rendered correctly

**Implementation Progress:** 53% (8/15 tasks)
