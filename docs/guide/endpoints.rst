===============
ADMET Endpoints
===============

The OpenADMET Challenge predicts 9 clinically-relevant ADMET properties that
characterize drug-like behavior. Understanding these endpoints helps interpret
model predictions and design better molecules.

.. mermaid::

   flowchart TB
      subgraph "💊 Drug Molecule"
         DRUG((Compound))
      end

      subgraph "🧬 Absorption"
         A1[Log KSOL<br/>Solubility]
         A2[Caco-2 Papp<br/>Permeability]
         A3[Caco-2 Efflux<br/>Transport]
      end

      subgraph "🩸 Distribution"
         D1[LogD<br/>Lipophilicity]
         D2[MPPB<br/>Plasma Binding]
         D3[MBPB<br/>Brain Binding]
         D4[MGMB<br/>Brain/Plasma]
      end

      subgraph "⚗️ Metabolism"
         M1[HLM CLint<br/>Human Liver]
         M2[MLM CLint<br/>Mouse Liver]
      end

      DRUG --> A1
      DRUG --> A2
      DRUG --> A3
      DRUG --> D1
      DRUG --> D2
      DRUG --> D3
      DRUG --> D4
      DRUG --> M1
      DRUG --> M2

      style DRUG fill:#fff9c4,stroke:#f9a825
      style A1 fill:#c8e6c9
      style A2 fill:#c8e6c9
      style A3 fill:#c8e6c9
      style D1 fill:#e1f5fe
      style D2 fill:#e1f5fe
      style D3 fill:#e1f5fe
      style D4 fill:#e1f5fe
      style M1 fill:#f3e5f5
      style M2 fill:#f3e5f5

Endpoint Summary
================

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Endpoint
     - Units
     - Description
   * - LogD
     - log units
     - Distribution coefficient at pH 7.4
   * - Log KSOL
     - log(μM)
     - Aqueous solubility
   * - Log HLM CLint
     - log(μL/min/mg)
     - Human liver microsomal clearance
   * - Log MLM CLint
     - log(μL/min/mg)
     - Mouse liver microsomal clearance
   * - Log Caco-2 Papp A>B
     - log(10⁻⁶ cm/s)
     - Intestinal permeability (apical to basolateral)
   * - Log Caco-2 Efflux
     - log ratio
     - Efflux ratio (B>A / A>B)
   * - Log MPPB
     - log(% bound)
     - Mouse plasma protein binding
   * - Log MBPB
     - log(% bound)
     - Mouse brain plasma binding
   * - Log MGMB
     - log ratio
     - Mouse brain-to-plasma ratio

Absorption
==========

Log KSOL — Aqueous Solubility
-----------------------------

**What it measures**: How well a compound dissolves in water at physiological pH.

**Units**: log(μM) — log of micromolar concentration

**Why it matters**:

- Oral drugs must dissolve in GI tract fluids for absorption
- Poor solubility (< 10 μM) often limits bioavailability
- Formulation strategies (salts, nanoparticles) can improve low solubility

**Interpretation**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value Range
     - Interpretation
   * - > 2 (> 100 μM)
     - High solubility — favorable for oral dosing
   * - 1 to 2 (10-100 μM)
     - Moderate — may need formulation optimization
   * - < 1 (< 10 μM)
     - Low solubility — likely bioavailability issues

Log Caco-2 Papp A>B — Intestinal Permeability
---------------------------------------------

**What it measures**: Rate of passive transport across intestinal epithelium.

**Units**: log(10⁻⁶ cm/s) — log of permeability coefficient

**Why it matters**:

- Predicts oral absorption through intestinal wall
- Caco-2 cells mimic human intestinal epithelium
- A>B direction represents absorption (gut → blood)

**Interpretation**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value Range
     - Interpretation
   * - > 1 (> 10 × 10⁻⁶ cm/s)
     - High permeability — good absorption expected
   * - 0 to 1
     - Moderate permeability
   * - < 0 (< 1 × 10⁻⁶ cm/s)
     - Low permeability — poor absorption

Log Caco-2 Efflux — P-gp Efflux
-------------------------------

**What it measures**: Ratio of basolateral-to-apical vs apical-to-basolateral transport.

**Units**: log(B>A / A>B ratio)

**Why it matters**:

- High efflux indicates active P-glycoprotein (P-gp) transport
- P-gp pumps drugs out of cells, reducing absorption
- Efflux ratio > 2 suggests P-gp substrate

**Interpretation**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value Range
     - Interpretation
   * - < 0.3
     - No/low efflux — not a P-gp substrate
   * - 0.3 to 0.5
     - Moderate efflux
   * - > 0.5 (ratio > 3)
     - High efflux — likely P-gp substrate

Distribution
============

LogD — Lipophilicity
--------------------

**What it measures**: Partition between octanol and water at pH 7.4.

**Units**: log units (unitless ratio)

**Why it matters**:

- Affects membrane permeability, protein binding, metabolism
- Central property influencing most ADMET parameters
- Optimal range for oral drugs: 1-3

**Interpretation**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value Range
     - Interpretation
   * - < 0
     - Hydrophilic — poor membrane permeability
   * - 1 to 3
     - Optimal for oral drugs
   * - > 5
     - Highly lipophilic — toxicity/metabolism risks

Log MPPB — Mouse Plasma Protein Binding
---------------------------------------

**What it measures**: Fraction of drug bound to plasma proteins in mouse.

**Units**: log(% bound)

**Why it matters**:

- Only unbound drug is pharmacologically active
- High binding reduces free drug concentration
- Affects half-life and distribution

**Interpretation**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value Range
     - Interpretation
   * - < 1.7 (< 50% bound)
     - Low binding — more free drug
   * - 1.7 to 1.95 (50-90%)
     - Moderate binding
   * - > 1.95 (> 90%)
     - High binding — potential displacement issues

Log MBPB — Mouse Brain Protein Binding
--------------------------------------

**What it measures**: Protein binding in mouse brain tissue.

**Units**: log(% bound)

**Why it matters**:

- Relevant for CNS drugs
- Brain-specific binding differs from plasma
- Affects brain free drug concentration

Log MGMB — Brain-to-Plasma Ratio
--------------------------------

**What it measures**: Ratio of drug concentration in brain vs plasma.

**Units**: log(Kp,brain)

**Why it matters**:

- Predicts CNS penetration
- Important for CNS drugs (want high) and peripherally-acting drugs (want low)
- Affected by P-gp at blood-brain barrier

**Interpretation**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value Range
     - Interpretation
   * - < -1 (< 0.1)
     - Poor CNS penetration
   * - -1 to 0 (0.1-1)
     - Moderate penetration
   * - > 0 (> 1)
     - Good CNS penetration

Metabolism
==========

Log HLM CLint — Human Liver Microsomal Clearance
------------------------------------------------

**What it measures**: Rate of metabolic degradation by human liver enzymes.

**Units**: log(μL/min/mg protein)

**Why it matters**:

- Primary determinant of oral bioavailability
- High clearance = short half-life
- CYP450 enzymes are primary metabolizers

**Interpretation**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value Range
     - Interpretation
   * - < 1 (< 10 μL/min/mg)
     - Low clearance — good metabolic stability
   * - 1 to 2 (10-100)
     - Moderate clearance
   * - > 2 (> 100)
     - High clearance — rapid metabolism

Log MLM CLint — Mouse Liver Microsomal Clearance
------------------------------------------------

**What it measures**: Metabolic clearance in mouse liver microsomes.

**Units**: log(μL/min/mg protein)

**Why it matters**:

- Preclinical prediction of metabolic stability
- Mouse is common preclinical species
- Species differences affect translation to human

Multi-Task Relationships
========================

These endpoints are correlated through common molecular features:

**Lipophilicity cluster** (driven by LogD):

- LogD ↔ MPPB ↔ MBPB (high correlation)
- LogD ↔ Caco-2 Papp (moderate correlation)

**Metabolism cluster**:

- HLM CLint ↔ MLM CLint (high correlation across species)
- LogD → CLint (lipophilic compounds often more metabolized)

**Permeability cluster**:

- Caco-2 Papp ↔ Efflux (inverse relationship)
- Caco-2 Papp ↔ KSOL (both affect absorption)

Multi-task learning exploits these relationships to improve predictions, especially
for endpoints with limited data.

Data Quality Considerations
===========================

The dataset includes quality annotations for each measurement:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Quality
     - Description
   * - High
     - Primary experimental data with good reproducibility
   * - Medium
     - Data from literature or with moderate uncertainty
   * - Low
     - Predicted values or high-uncertainty measurements

Curriculum learning uses these quality labels to train models progressively
on cleaner data first. See :doc:`curriculum` for details.

.. seealso::

   - :doc:`data_sources` — Data loading and preprocessing
   - :doc:`curriculum` — Quality-aware curriculum learning
   - :doc:`modeling` — Model training workflows
