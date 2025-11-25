"""Tests for FeaturizationMethod enum behavior.

Validates enum values and instance types.
"""

from __future__ import annotations

import pytest

from admet.train.base import FeaturizationMethod


@pytest.mark.unit
def test_featurization_enum_members() -> None:
    assert FeaturizationMethod.MORGAN_FP == "morgan_fp"
    assert FeaturizationMethod.SMILES == "smiles"
    assert FeaturizationMethod.NONE == "none"


@pytest.mark.unit
def test_featurization_instance_and_type() -> None:
    fm: FeaturizationMethod = FeaturizationMethod.MORGAN_FP
    assert isinstance(fm, FeaturizationMethod)
    assert fm == "morgan_fp"
