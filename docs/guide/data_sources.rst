Data Sources & Curation
=======================

This guide summarizes the origin, preprocessing, and quality categorization
of datasets used in the OpenADMET + ExpansionRx Blind Challenge.

Source Inventory
----------------

Raw inputs are organized under `assets/raw/` by provider/source.
Common sources include (illustrative list):

- Admetica (`assets/raw/admetica/`)
- ChEMBL (`assets/raw/ChEMBL/`)
- ExpansionRx internal challenge sets (`assets/raw/ExpansionRX/`)
- KERMT (`assets/raw/KERMT/`)
- NCATS (`assets/raw/NCATS/`)
- PharmaBench (`assets/raw/PharmaBench/`)
- Polaris (Antiviral / Biogen) (`assets/raw/Polaris-*`)
- TDC (`assets/raw/TDC/`)

Target Endpoints
----------------

The challenge focuses on the following ADMET endpoints:

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Column
     - Unit
     - Type
     - Description
   * - Molecule Name
     -
     - str
     - Identifier for the molecule
   * - Smiles
     -
     - str
     - Text representation of the 2D molecular structure
   * - LogD
     -
     - float
     - LogD calculation
   * - KSol
     - uM
     - float
     - Kinetic Solubility
   * - MLM CLint
     - mL/min/kg
     - float
     - Mouse Liver Microsomal Clearance
   * - HLM CLint
     - mL/min/kg
     - float
     - Human Liver Microsomal Clearance
   * - Caco-2 Permeability Efflux
     -
     - float
     - Caco-2 Permeability Efflux Ratio
   * - Caco-2 Permeability Papp A>B
     - 10^-6 cm/s
     - float
     - Caco-2 Permeability (Apical to Basolateral)
   * - MPPB
     - % Unbound
     - float
     - Mouse Plasma Protein Binding
   * - MBPB
     - % Unbound
     - float
     - Mouse Brain Protein Binding
   * - MGMB
     - % Unbound
     - float
     - Mouse Gastrocnemius Muscle Binding

Curation Steps (High-Level)
---------------------------

Each raw dataset typically undergoes:

1. Schema Normalization: Standardize column names (e.g. SMILES, target, label/value).
2. Type Coercion: Convert numerical columns; ensure categorical consistency.
3. Chemical Standardization: Remove salts, canonicalize SMILES (tooling invoked in `chem.py`).
4. Duplicate Handling: Drop exact duplicates; optionally average replicate measures.
5. Quality Annotation: Assign quality tier (high / medium) based on predefined heuristics.
6. Merge / Union: Combine curated sources into a unified table.
7. Export: Write cleaned artifacts to `assets/dataset/eda/data/` (e.g. `cleaned_combined_datasets.csv`).

Quality Tiers
-------------

Two primary tiers currently referenced:

- High Quality: Reliable measurements, consistent assay conditions, minimal missingness.
- Medium Quality: Useful but noisier measurements or partial metadata.

(Exact heuristics should be documented here when finalized—placeholder for now.)

Resulting Artifacts
-------------------

Key curated outputs:

- `cleaned_combined_datasets.csv`: Consolidated cleaned dataset across sources.
- `cleaned_datasets_quality_table.csv`: Per‑source quality indicators and summary stats.
- Split directories under `assets/raw/splits/` and `assets/dataset/splits/` for partitioned sets.

Data Provenance & Traceability
------------------------------

Provenance is maintained by retaining source‑specific subdirectories and optional
`curation.md` narrative files. When merging, a `source` column or similar indicator
should allow reverse lookup of origin.

Reproducibility Practices
-------------------------

- Record version or snapshot date of external datasets in a manifest.
- Pin transformation code (no ad‑hoc manual edits to CSVs).
- Maintain deterministic ordering when merging (e.g. sort by SMILES then source).

Planned Enhancements
--------------------

- Formal manifest file enumerating each dataset with checksum.
- Automated validation script verifying expected columns and distributions.
- Expanded set of quality tiers (e.g. experimental vs inferred).

Cross-References
----------------

- See :doc:`splitting` for how quality tiers influence train/validation/test partitioning.
- See :doc:`architecture` for where curation lives in the broader system.

