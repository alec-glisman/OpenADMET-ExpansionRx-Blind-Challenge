"""Top-level admet package.

This module implements a lazy-importing façade: heavy submodules (RDKit,
seaborn, etc.) live under ``admet.data`` and ``admet.visualize``. Accessing
helpers from ``admet`` will import the respective submodule on first use,
keeping ``import admet`` fast.

Public API names are listed in ``__all__`` and resolved lazily via
``__getattr__`` (PEP 562).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

# Mapping of attribute name -> (module_path, attribute_name)
_LAZY_MAP: Dict[str, tuple[str, str]] = {
    # chem helpers (implemented under admet.data.chem)
    "canonicalize_smiles": ("admet.data.chem", "canonicalize_smiles"),
    "parallel_canonicalize_smiles": ("admet.data.chem", "parallel_canonicalize_smiles"),
    "compute_molecular_properties": ("admet.data.chem", "compute_molecular_properties"),
    # dataset/constants (under admet.data.constants)
    "COLS_WITH_UNITS": ("admet.data.constants", "COLS_WITH_UNITS"),
    "TRANSFORMATIONS": ("admet.data.constants", "TRANSFORMATIONS"),
    "cols_with_units": ("admet.data.constants", "cols_with_units"),
    "transformations": ("admet.data.constants", "transformations"),
    # plotting utilities (under admet.visualize.plots)
    "calc_stats": ("admet.visualize.plots", "calc_stats"),
    "plot_numeric_distributions": ("admet.visualize.plots", "plot_numeric_distributions"),
    "plot_correlation_matrix": ("admet.visualize.plots", "plot_correlation_matrix"),
    "plot_property_distributions": ("admet.visualize.plots", "plot_property_distributions"),
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    """Lazily import attributes defined in ``_LAZY_MAP``.

    On first access the target submodule is imported and the attribute is
    cached in this module's globals so subsequent access is fast and does not
    re-import.
    """
    if name in _LAZY_MAP:
        module_name, attr_name = _LAZY_MAP[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'admet' has no attribute '{name}'")


def __dir__() -> List[str]:
    """Return module attributes plus lazily exportable names."""
    return sorted(list(globals().keys()) + list(_LAZY_MAP.keys()))


__all__ = list(_LAZY_MAP.keys())
