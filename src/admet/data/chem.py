"""Cheminformatics utility functions.

This module provides small, reusable helpers used in notebooks and scripts for
SMILES canonicalization, parallel processing, and computing common molecular
properties. RDKit is used for molecule handling.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent import futures
from typing import Iterable, List, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, SaltRemover, rdMolDescriptors
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
_SALT_REMOVER = SaltRemover.SaltRemover()


def canonicalize_smiles(smiles: str, isomeric: bool = True) -> Optional[str]:
    """Convert a SMILES string to canonical form with defensive checks.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    isomeric : bool, optional
        Whether to keep isomeric information, by default True.

    Returns
    -------
    Optional[str]
        Canonical SMILES string, or None if parsing fails or results in an
        empty molecule.
    """
    if smiles is None:
        return None
    s = str(smiles)

    mol = getattr(Chem, "MolFromSmiles")(s)
    if mol is None:
        logger.debug("Invalid SMILES string could not be parsed: %s", s)
        return None

    try:
        mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    except (ValueError, RuntimeError):  # pragma: no cover - very rare RDKit error path
        logger.exception("Salt removal failed for SMILES: %s", s)
        return None

    if mol.GetNumAtoms() == 0:
        logger.debug("Molecule empty after salt removal: %s", s)
        return None

    smi = getattr(Chem, "MolToSmiles")(mol, canonical=True, isomericSmiles=isomeric, doRandom=False)
    return smi


def parallel_canonicalize_smiles(
    smiles_list: Iterable, isomeric: bool = True, max_workers: int | None = None
) -> List[Optional[str]]:
    """Canonicalize a list/iterable of SMILES in parallel.

    The output order is stable and corresponds to the input order. Errors are
    logged and result in None for the corresponding element.

    Parameters
    ----------
    smiles_list : Iterable
        Iterable of SMILES to canonicalize.
    isomeric : bool, optional
        Preserve isomeric information, by default True.
    max_workers : int | None, optional
        Maximum worker threads. If None, a value based on CPU count is used.

    Returns
    -------
    List[Optional[str]]
        List of canonical SMILES with None where canonicalization failed.
    """
    if smiles_list is None:
        return []

    smiles_seq = list(smiles_list)
    n = len(smiles_seq)
    if n == 0:
        return []

    if max_workers is None:
        # Similar heuristic as concurrent.futures default for ThreadPool
        try:
            cpu = mp.cpu_count() or 1
        except NotImplementedError:
            cpu = 1
        max_workers = min(32, cpu + 4)

    logger.debug(
        "Using %d workers for parallel SMILES canonicalization. Total tasks: %d",
        max_workers,
        n,
    )

    results: List[Optional[str]] = [None] * n
    future_to_index = {}
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, smi in enumerate(smiles_seq):
            future = executor.submit(canonicalize_smiles, smi, isomeric)
            future_to_index[future] = idx

        for completed in tqdm(
            futures.as_completed(future_to_index),
            total=n,
            desc="Canonicalizing SMILES",
        ):
            idx = future_to_index[completed]
            exc = completed.exception()
            if exc is not None:
                logger.exception(
                    "Canonicalization task failed for index %d (SMILES: %r)",
                    idx,
                    smiles_seq[idx],
                )
                res = None
            else:
                res = completed.result()
            results[idx] = res

    return results


def compute_molecular_properties(smiles: Iterable[str]) -> pd.DataFrame:
    """Compute common molecular properties for an iterable of SMILES strings.

    Parameters
    ----------
    smiles : Iterable[str]
        Iterable of SMILES strings.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: [SMILES, MW, TPSA, HBA, HBD, RotBonds, LogP,
        NumHeavyAtoms, NumRings]. Invalid SMILES are skipped.
    """
    rows: list[dict] = []
    series = pd.Series(list(smiles)).dropna().astype(str)
    for smi in series.tolist():
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


__all__ = [
    "canonicalize_smiles",
    "parallel_canonicalize_smiles",
    "compute_molecular_properties",
]
