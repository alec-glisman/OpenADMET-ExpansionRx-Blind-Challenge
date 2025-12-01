"""admet.data.chem
===================

Cheminformatics utility helpers built on top of RDKit.

This module centralises lightweight, reusable functionality for:

* Canonicalising SMILES strings with salt removal (``canonicalize_smiles``)
* Parallel canonicalisation over an iterable (``parallel_canonicalize_smiles``)
* Computing common physico‑chemical properties (``compute_molecular_properties``)

The functions are intentionally defensive: parsing failures, empty molecules
after salt stripping, and unexpected RDKit errors are converted into ``None``
entries (or skipped rows) instead of raising, so that downstream batch
processing code can proceed.

Examples
--------
Canonicalise a list of SMILES and compute properties::

    from admet.data.chem import parallel_canonicalize_smiles, compute_molecular_properties

    raw = ["CC(=O)Oc1ccccc1C(=O)O", "invalid", "C1=CC=CN=C1"]
    clean = parallel_canonicalize_smiles(raw)
    props_df = compute_molecular_properties([s for s in clean if s])
    print(props_df.head())

Threading Model
---------------
``parallel_canonicalize_smiles`` uses a thread pool (rather than processes)
because RDKit releases the GIL in many descriptor / parsing operations, and
thread pools have lower overhead for the short tasks performed here.

RDKit Dependency
----------------
All functionality relies on RDKit. Minimal attribute access via ``getattr`` is
used to avoid mypy import-time stubs mismatches and to make monkey‑patching
easier during certain types of testing.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent import futures
from typing import Iterable, List, Optional

from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import SaltRemover  # type: ignore[import-not-found]
from tqdm.auto import tqdm  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)
_SALT_REMOVER = SaltRemover.SaltRemover()


def canonicalize_smiles(smiles: str, isomeric: bool = True) -> Optional[str]:
    """Canonicalise a single SMILES string with salt removal.

    The function performs parsing, removes salts, checks for an empty
    molecule, and converts back to a canonical SMILES string. Failures are
    logged at DEBUG level and returned as ``None``.

    Parameters
    ----------
    smiles : str
        Input raw SMILES string; ``None`` returns ``None`` immediately.
    isomeric : bool, optional
        Preserve isomeric information (``isomericSmiles``), by default ``True``.

    Returns
    -------
    Optional[str]
        Canonical SMILES string or ``None`` if parsing / cleaning fails or the
        molecule becomes empty after salt stripping.

    Raises
    ------
    None
        All errors are caught and converted to ``None``; unexpected RDKit
        exceptions during salt removal are logged and suppressed.
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
    """Canonicalise many SMILES strings concurrently.

    A thread pool is used with a heuristic for worker count (``min(32, cpu+4)``)
    similar to the default used by ``concurrent.futures``. Ordering of the
    output list matches the input ordering exactly.

    Parameters
    ----------
    smiles_list : Iterable
        Iterable of raw SMILES strings.
    isomeric : bool, optional
        Preserve isomeric information, by default ``True``.
    max_workers : int | None, optional
        Explicit thread count; ``None`` applies an automatic heuristic.

    Returns
    -------
    List[Optional[str]]
        Canonical SMILES strings, with ``None`` for entries that failed.

    Raises
    ------
    None
        All underlying errors are caught; per‑item failures produce ``None``.
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
