# Data module

## Endpoints

| Column                       | Unit        | Type      | Description                                   |
|:---------------------------- |:----------: |:--------: |:----------------------------------------------|
| Molecule Name                |             |    str    | Identifier for the molecule |
| Smiles                       |             |    str    | Text representation of the 2D molecular structure |
| LogD                         |             |   float   | LogD calculation |
| KSol                         |    uM       |   float   | Kinetic Solubility |
| MLM CLint                    | mL/min/kg   |   float   | Mouse Liver Microsomal |
| HLM CLint                    | mL/min/kg   |   float   | Human Liver Microsomal |
| Caco-2 Permeability Efflux   |             |   float   | Caco-2 Permeability Efflux |
| Caco-2 Permeability Papp A>B | 10^-6 cm/s  |   float   | Caco-2 Permeability Papp A>B |
| MPPB                         | % Unbound   |   float   | Mouse Plasma Protein Binding |
| MBPB                         | % Unbound   |   float   | Mouse Brain Protein Binding |
| MGMB.                        | % Unbound   |   float   | Mouse Gastrocnemius Muscle Binding |

### Questions

- [ ] What pH is used for LogD measurements?
- [ ] What pH is used for KSol measurements?

## External Datasets

- [ ] [KERMT](https://figshare.com/articles/dataset/Datasets_for_Multitask_finetuning_and_acceleration_of_chemical_pretrained_models_for_small_molecule_drug_property_prediction_/30350548/2)
- [ ] [Polaris Antiviral](https://polarishub.io/datasets/asap-discovery/antiviral-admet-2025-unblinded)
- [ ] [Polaris ADME Fang](https://polarishub.io/datasets/biogen/adme-fang-v1)
- [ ] [TDC](https://tdcommons.ai/benchmark/admet_group/overview/)
- [ ] [PharmaBench](https://github.com/mindrank-ai/PharmaBench)
- [ ] [NCATS](https://opendata.ncats.nih.gov/adme/data)
- [ ] [admetSAR 3.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC11223829/#:~:text=Data%20collection,are%20available%20in%20Text%20S2.)
- [ ] [admetics](https://github.com/datagrok-ai/admetica)

### [KERMT](https://figshare.com/articles/dataset/Datasets_for_Multitask_finetuning_and_acceleration_of_chemical_pretrained_models_for_small_molecule_drug_property_prediction_/30350548/2)

#### Summary

TODO

- Dataset size:
- Endpoints included:
  - CL_microsome_human
  - CL_microsome_mouse
  - CL_microsome_rat
  - CL_total_dog
  - CL_total_human
  - CL_total_monkey
  - CL_total_rat
  - CYP2C8_inhibition
  - CYP2C9_inhibition
  - CYP2D6_inhibition
  - CYP3A4_inhibition
  - Dog_fraction_unbound_plasma
  - Human_fraction_unbound_plasma
  - Monkey_fraction_unbound_plasma
  - Rat_fraction_unbound_plasma
  - Papp_Caco2
  - Pgp_human
  - hERG_binding
  - **LogD_pH_7.4**
  - kinetic_logSaq
  - thermo_logSaq
  - VDss_dog
  - VDss_human
  - VDss_monkey
  - VDss_rat
