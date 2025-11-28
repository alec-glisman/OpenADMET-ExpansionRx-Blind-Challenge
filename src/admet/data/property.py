from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors  # type: ignore[import-not-found]
from tqdm.auto import tqdm  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def compute_molecular_properties(smiles: Iterable[str]) -> pd.DataFrame:
    """Compute common physico‑chemical properties for SMILES strings.

    Invalid SMILES rows are skipped silently. The function returns a tidy
    DataFrame suitable for merging back onto an existing table by SMILES.

    Parameters
    ----------
    smiles : Iterable[str]
        Iterable of (possibly repeated) SMILES strings.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``[SMILES, MW, TPSA, HBA, HBD, RotBonds, LogP,
        NumHeavyAtoms, NumRings]``.

    Raises
    ------
    None
        Parsing errors are suppressed; problematic SMILES are omitted.
    """
    rows: list[dict] = []
    series = pd.Series(list(smiles)).dropna().astype(str)
    for smi in tqdm(series.tolist(), desc="Computing molecular properties", unit="molecule"):
        mol = getattr(Chem, "MolFromSmiles")(smi)
        if mol is None:
            continue
        rows.append(
            {
                "SMILES": smi,
                "MW": getattr(Descriptors, "MolWt")(mol),
                "TPSA": getattr(Descriptors, "TPSA")(mol),
                "HBA": getattr(Lipinski, "NumHAcceptors")(mol),
                "HBD": getattr(Lipinski, "NumHDonors")(mol),
                "RotBonds": getattr(Descriptors, "NumRotatableBonds")(mol),
                "LogP": getattr(Descriptors, "MolLogP")(mol),
                "NumHeavyAtoms": getattr(Descriptors, "HeavyAtomCount")(mol),
                "NumRings": rdMolDescriptors.CalcNumRings(mol),
            }
        )
    return pd.DataFrame(rows)
