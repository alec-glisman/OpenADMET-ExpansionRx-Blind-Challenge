"""Sphinx configuration for OpenADMET Challenge docs.

This configuration enables autodoc, napoleon (NumPy docstrings), MyST for
Markdown, and type hint processing. The build expects the package source in
``src/`` and adds that path to ``sys.path`` so modules can be imported.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Add project src directory to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

project = "OpenADMET Challenge"
author = "Alec Glisman, Ph.D."
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_panels",
    "sphinx_tabs.tabs",
    "sphinxext.opengraph",
]

# Generate autosummary stubs automatically (disabled while resolving recursion)
autosummary_generate = False

# Napoleon settings (NumPy style primarily)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# MyST parser options
myst_enable_extensions = ["deflist", "fieldlist", "colon_fence"]
# Suppress heading level warnings for planning docs (draft content)
suppress_warnings = ["myst.header", "autodoc.import_object"]

# Mock imports for modules that require compiled extensions or are not available during doc build
autodoc_mock_imports = ["bitbirch.pruning"]

# Type hints
autodoc_typehints = "description"

# Master document
master_doc = "index"

# HTML output options
# Use a modern, clean theme
html_theme = "furo"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_title = project
html_show_sourcelink = True
html_show_copyright = True

# Optional branding
html_logo = "_static/images/logo.svg"
html_favicon = "_static/images/logo.svg"

# Furo theme customization for brand colors
html_theme_options = {
    "navigation_with_keys": True,
    # "light_logo": "_static/images/logo.svg",
    # "dark_logo": "_static/images/logo.svg",
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#007A73",
        "color-brand-content": "#C4C7F2",
    },
    "dark_css_variables": {
        "color-brand-primary": "#007A73",
        "color-brand-content": "#C4C7F2",
    },
}

# OpenGraph: social card & metadata (helps link previews)
ogp_site_url = "https://github.com/alec-glisman/OpenADMET-ExpansionRx-Blind-Challenge"
ogp_description_length = 200
# Use project logo as default image (relative path copied to static)
ogp_image = html_logo
ogp_type = "website"

# Intersphinx mapping for common dependencies
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "xgboost": ("https://xgboost.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "ray": ("https://docs.ray.io/en/latest/", None),
    "lightgbm": ("https://lightgbm.readthedocs.io/en/latest/", None),
    # "mlflow": ("https://mlflow.org/docs/latest/", None),
    # "transformers": ("https://huggingface.co/docs/transformers/", None),
    # "datasets": ("https://huggingface.co/docs/datasets/", None),
}

# Exclusions
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Copybutton
copybutton_exclude = ".linenos, .gp, .go"
