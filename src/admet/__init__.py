"""
ADMET Prediction Package
========================

This package provides tools for training and evaluating machine learning models
for ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) property prediction.

.. module:: admet

"""

from __future__ import annotations

__version__ = "1.1.0"

# Leaderboard module for Gradio scraping and analysis
from admet import leaderboard

__all__ = ["leaderboard"]
