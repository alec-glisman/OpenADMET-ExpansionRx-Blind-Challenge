===============
Troubleshooting
===============

Solutions to common issues encountered during installation, training, and deployment.

.. mermaid::

   flowchart LR
      subgraph "🔴 Problem"
         A[Error<br/>Message]
      end

      subgraph "🔍 Diagnose"
         B[Check Logs]
         C[Identify<br/>Category]
      end

      subgraph "✅ Resolve"
         D[Installation]
         E[Training]
         F[HPO/Ray]
         G[MLflow]
      end

      A --> B --> C
      C --> D
      C --> E
      C --> F
      C --> G

      style A fill:#ffcdd2
      style D fill:#c8e6c9
      style E fill:#c8e6c9
      style F fill:#c8e6c9
      style G fill:#c8e6c9

Installation Issues
===================

Python Version Mismatch
-----------------------

**Error**: ``This package requires Python >=3.11``

**Solution**: Install Python 3.11:

.. code-block:: bash

   # macOS with Homebrew
   brew install python@3.11

   # Ubuntu/Debian
   sudo apt install python3.11 python3.11-venv

   # Verify
   python3.11 --version

BitBirch Import Error
---------------------

**Error**: ``ImportError: cannot import name 'pruning' from 'bitbirch'``

**Solution**: This module requires compilation. Core functionality works without it.
For full support:

.. code-block:: bash

   # Install build dependencies
   pip install cython numpy

   # Reinstall bitbirch
   pip install --no-cache-dir bitbirch

Training Issues
===============

CUDA Out of Memory
------------------

**Error**: ``CUDA out of memory. Tried to allocate X MiB``

**Solutions**:

1. **Reduce batch size**:

   .. code-block:: yaml

      training:
        batch_size: 32  # Try 16 or 8

2. **Share GPUs across trials** (for HPO):

   .. code-block:: yaml

      resources:
        gpus_per_trial: 0.5  # 2 trials per GPU

3. **Enable gradient checkpointing**:

   .. code-block:: yaml

      model:
        chemprop:
          gradient_checkpointing: true

4. **Use mixed precision**:

   .. code-block:: yaml

      training:
        precision: 16-mixed

NaN Loss During Training
------------------------

**Error**: ``Loss is NaN at epoch X``

**Solutions**:

1. **Lower learning rate**:

   .. code-block:: yaml

      training:
        learning_rate: 0.0001  # Was 0.001

2. **Add gradient clipping**:

   .. code-block:: yaml

      training:
        gradient_clip_val: 1.0

3. **Check for invalid SMILES** in your data:

   .. code-block:: python

      from rdkit import Chem
      invalid = [smi for smi in smiles_list if Chem.MolFromSmiles(smi) is None]
      print(f"Invalid SMILES: {len(invalid)}")

Slow Training
-------------

**Symptom**: Training takes much longer than expected

**Solutions**:

1. **Enable DataLoader workers**:

   .. code-block:: yaml

      training:
        num_workers: 4  # Match CPU cores

2. **Use persistent workers**:

   .. code-block:: yaml

      training:
        persistent_workers: true

3. **Check GPU utilization**:

   .. code-block:: bash

      nvidia-smi -l 1  # Monitor GPU usage

Ray and HPO Issues
==================

Ray Initialization Fails
------------------------

**Error**: ``ray.exceptions.RaySystemError: Could not find a running Ray instance``

**Solution**: Stop stale processes and restart:

.. code-block:: bash

   ray stop --force
   export RAY_ADDRESS=local

   # Then run HPO
   admet model hpo -c config.yaml

Trial Fails Immediately
-----------------------

**Error**: ``Trial failed after 0 iterations``

**Solutions**:

1. **Check config syntax**:

   .. code-block:: bash

      python -c "from omegaconf import OmegaConf; OmegaConf.load('config.yaml')"

2. **Run single trial first**:

   .. code-block:: bash

      admet model train -c config.yaml  # Debug with single run

3. **Check resource availability**:

   .. code-block:: yaml

      resources:
        cpus_per_trial: 2
        gpus_per_trial: 0.5  # Ensure you have enough

ASHA Not Stopping Bad Trials
----------------------------

**Symptom**: All trials run to completion, no early stopping

**Check**: Verify ASHA configuration:

.. code-block:: yaml

   asha:
     metric: val_mae          # Must match logged metric name
     mode: min                # "min" for loss/MAE, "max" for accuracy
     grace_period: 10         # Minimum epochs before stopping
     reduction_factor: 3

Optuna Study Not Found
----------------------

**Error**: ``Study 'study_name' not found in database``

**Solution**: List available studies:

.. code-block:: bash

   admet model hpo-list-studies --storage-dir hpo_results/optuna_studies

MLflow Issues
=============

Cannot Connect to MLflow
------------------------

**Error**: ``ConnectionError: Unable to connect to MLflow tracking server``

**Solution**: Start the server:

.. code-block:: bash

   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

Or use local file tracking:

.. code-block:: bash

   export MLFLOW_TRACKING_URI=./mlruns

Artifacts Not Logging
---------------------

**Symptom**: Runs appear but artifacts are missing

**Check**: Verify artifact root is writable:

.. code-block:: bash

   ls -la mlruns/
   # Should show write permissions

Run ID Not Found
----------------

**Error**: ``MlflowException: Run 'abc123' not found``

**Solution**: The run may have been deleted. List recent runs:

.. code-block:: bash

   mlflow runs list --experiment-name "your_experiment"

Data Issues
===========

Invalid SMILES
--------------

**Error**: ``RDKit: Cannot parse molecule from SMILES``

**Solution**: Validate and clean SMILES:

.. code-block:: python

   from rdkit import Chem

   def clean_smiles(smi):
       mol = Chem.MolFromSmiles(smi)
       if mol is None:
           return None
       return Chem.MolToSmiles(mol, canonical=True)

   df['SMILES'] = df['SMILES'].apply(clean_smiles)
   df = df.dropna(subset=['SMILES'])

Missing Target Columns
----------------------

**Error**: ``KeyError: 'LogD' not in dataframe``

**Solution**: Check column names match exactly:

.. code-block:: python

   print(df.columns.tolist())
   # Verify target_columns in config match

Empty Validation Set
--------------------

**Error**: ``ValueError: Validation set is empty``

**Solution**: Ensure split produces both train and val data:

.. code-block:: bash

   admet data split data.csv --val-size 0.1 --test-size 0.1

Configuration Issues
====================

YAML Syntax Error
-----------------

**Error**: ``yaml.scanner.ScannerError: mapping values are not allowed``

**Solution**: Check indentation (2 spaces, no tabs):

.. code-block:: yaml

   # Wrong
   model:
   chemprop:  # Missing indent
       depth: 4

   # Correct
   model:
     chemprop:
       depth: 4

Unknown Config Key
------------------

**Error**: ``ConfigAttributeError: Key 'unknown_key' not in config``

**Solution**: Check spelling and valid options in :doc:`config_reference`.

Ensemble Issues
===============

Checkpoint Not Found
--------------------

**Error**: ``FileNotFoundError: Checkpoint at split_0/fold_0/best.ckpt not found``

**Solution**: Ensure training completed successfully:

.. code-block:: bash

   # Check for checkpoints
   find . -name "*.ckpt" -type f

   # Verify training logs
   cat split_0/fold_0/training.log

Memory Error During Ensemble
----------------------------

**Error**: ``MemoryError`` during ensemble prediction

**Solution**: Process in batches:

.. code-block:: python

   # Reduce inference batch size
   predictions = model.predict(data, batch_size=32)

Performance Tips
================

Speed Up Training
-----------------

1. **Use uv instead of pip** — 10-100x faster package installation
2. **Enable DataLoader workers** — Parallel data loading
3. **Use mixed precision** — 2x speedup on modern GPUs
4. **Reduce logging frequency** — Less I/O overhead

Reduce Memory Usage
-------------------

1. **Gradient checkpointing** — Trade compute for memory
2. **Smaller batch sizes** — Obvious but effective
3. **Model pruning** — Remove unused parameters
4. **FP16 training** — Half the memory for activations

Getting Help
============

If issues persist:

1. **Check logs**: ``cat training.log`` or MLflow UI
2. **Search issues**: `GitHub Issues <https://github.com/alec-glisman/OpenADMET-ExpansionRx-Blind-Challenge/issues>`_
3. **Open new issue**: Include error message, config, and environment info

.. seealso::

   - :doc:`/getting-started/installation` — Installation guide
   - :doc:`configuration` — Configuration reference
   - :doc:`profiling` — Performance profiling
