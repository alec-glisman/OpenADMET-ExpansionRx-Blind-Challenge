"""Tests for fingerprint generation module."""

from __future__ import annotations

import numpy as np
import pytest

from admet.features.fingerprints import FingerprintGenerator
from admet.model.config import FingerprintConfig

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def valid_smiles() -> list[str]:
    """Return list of valid SMILES strings."""
    return [
        "CCO",  # Ethanol
        "CCCO",  # Propanol
        "c1ccccc1",  # Benzene
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]


@pytest.fixture
def invalid_smiles() -> list[str]:
    """Return list of invalid SMILES strings."""
    return [
        "invalid_smiles",
        "not_a_molecule",
        "",
    ]


@pytest.fixture
def mixed_smiles(valid_smiles: list[str], invalid_smiles: list[str]) -> list[str]:
    """Return mix of valid and invalid SMILES."""
    return [valid_smiles[0], invalid_smiles[0], valid_smiles[1]]


# ============================================================================
# Morgan Fingerprint Tests
# ============================================================================


class TestMorganFingerprints:
    """Tests for Morgan (circular) fingerprints."""

    def test_default_config(self, valid_smiles: list[str]):
        """Test Morgan fingerprints with default configuration."""
        config = FingerprintConfig(type="morgan")
        gen = FingerprintGenerator(config)

        fps = gen.generate(valid_smiles)

        assert fps.shape == (len(valid_smiles), 2048)
        assert fps.dtype == np.int64 or fps.dtype == np.float64

    def test_custom_bits(self, valid_smiles: list[str]):
        """Test Morgan fingerprints with custom bit count."""
        config = FingerprintConfig(type="morgan")
        config.morgan.n_bits = 1024
        gen = FingerprintGenerator(config)

        fps = gen.generate(valid_smiles)

        assert fps.shape == (len(valid_smiles), 1024)

    def test_custom_radius(self, valid_smiles: list[str]):
        """Test Morgan fingerprints with different radii produce different results."""
        config1 = FingerprintConfig(type="morgan")
        config1.morgan.radius = 2
        gen1 = FingerprintGenerator(config1)

        config2 = FingerprintConfig(type="morgan")
        config2.morgan.radius = 3
        gen2 = FingerprintGenerator(config2)

        fps1 = gen1.generate(valid_smiles)
        fps2 = gen2.generate(valid_smiles)

        # Different radii should produce different fingerprints
        assert not np.array_equal(fps1, fps2)

    def test_chirality_option(self, valid_smiles: list[str]):
        """Test that chirality option is accepted."""
        config = FingerprintConfig(type="morgan")
        config.morgan.use_chirality = True
        gen = FingerprintGenerator(config)

        fps = gen.generate(valid_smiles)

        assert fps.shape == (len(valid_smiles), 2048)

    def test_fingerprint_dim_property(self):
        """Test fingerprint_dim property returns correct value."""
        config = FingerprintConfig(type="morgan")
        config.morgan.n_bits = 512
        gen = FingerprintGenerator(config)

        assert gen.fingerprint_dim == 512


# ============================================================================
# RDKit Fingerprint Tests
# ============================================================================


class TestRDKitFingerprints:
    """Tests for RDKit (path-based) fingerprints."""

    def test_default_config(self, valid_smiles: list[str]):
        """Test RDKit fingerprints with default configuration."""
        config = FingerprintConfig(type="rdkit")
        gen = FingerprintGenerator(config)

        fps = gen.generate(valid_smiles)

        assert fps.shape == (len(valid_smiles), 2048)

    def test_custom_bits(self, valid_smiles: list[str]):
        """Test RDKit fingerprints with custom bit count."""
        config = FingerprintConfig(type="rdkit")
        config.rdkit.n_bits = 1024
        gen = FingerprintGenerator(config)

        fps = gen.generate(valid_smiles)

        assert fps.shape == (len(valid_smiles), 1024)

    def test_custom_path_length(self, valid_smiles: list[str]):
        """Test RDKit fingerprints with custom path lengths."""
        config = FingerprintConfig(type="rdkit")
        config.rdkit.min_path = 2
        config.rdkit.max_path = 5
        gen = FingerprintGenerator(config)

        fps = gen.generate(valid_smiles)

        assert fps.shape == (len(valid_smiles), 2048)


# ============================================================================
# MACCS Keys Tests
# ============================================================================


class TestMACCSFingerprints:
    """Tests for MACCS keys."""

    def test_maccs_keys(self, valid_smiles: list[str]):
        """Test MACCS keys have correct shape (167 bits)."""
        config = FingerprintConfig(type="maccs")
        gen = FingerprintGenerator(config)

        fps = gen.generate(valid_smiles)

        assert fps.shape == (len(valid_smiles), 167)

    def test_fingerprint_dim_property(self):
        """Test fingerprint_dim is 167 for MACCS."""
        config = FingerprintConfig(type="maccs")
        gen = FingerprintGenerator(config)

        assert gen.fingerprint_dim == 167


# ============================================================================
# Mordred Descriptor Tests
# ============================================================================


class TestMordredDescriptors:
    """Tests for Mordred molecular descriptors."""

    @pytest.mark.slow
    def test_mordred_descriptors(self, valid_smiles: list[str]):
        """Test Mordred descriptors generation."""
        pytest.importorskip("mordred")

        config = FingerprintConfig(type="mordred")
        gen = FingerprintGenerator(config)

        # Just test first 2 molecules (mordred is slow)
        fps = gen.generate(valid_smiles[:2])

        # Mordred generates ~1800+ descriptors
        assert fps.shape[0] == 2
        assert fps.shape[1] > 1500

    @pytest.mark.slow
    def test_mordred_normalization(self, valid_smiles: list[str]):
        """Test Mordred normalization replaces NaN/inf values."""
        pytest.importorskip("mordred")

        config = FingerprintConfig(type="mordred")
        config.mordred.normalize = True
        gen = FingerprintGenerator(config)

        fps = gen.generate(valid_smiles[:2])

        # Should not contain NaN or inf after normalization
        assert not np.any(np.isnan(fps))
        assert not np.any(np.isinf(fps))


# ============================================================================
# Invalid SMILES Handling Tests
# ============================================================================


class TestInvalidSMILESHandling:
    """Tests for handling invalid SMILES."""

    def test_invalid_smiles_returns_zeros(self, invalid_smiles: list[str]):
        """Test that invalid SMILES return zero vectors."""
        config = FingerprintConfig(type="morgan")
        gen = FingerprintGenerator(config)

        fps = gen.generate(invalid_smiles)

        assert fps.shape == (len(invalid_smiles), 2048)
        # All fingerprints should be zeros
        assert np.allclose(fps, 0)

    def test_mixed_valid_invalid(self, mixed_smiles: list[str]):
        """Test mix of valid and invalid SMILES."""
        config = FingerprintConfig(type="morgan")
        gen = FingerprintGenerator(config)

        fps = gen.generate(mixed_smiles)

        assert fps.shape == (len(mixed_smiles), 2048)
        # First and third should be non-zero (valid)
        assert fps[0].sum() > 0
        assert fps[2].sum() > 0
        # Second should be zero (invalid)
        assert fps[1].sum() == 0

    def test_single_invalid_smiles(self):
        """Test generate_single with invalid SMILES."""
        config = FingerprintConfig(type="morgan")
        gen = FingerprintGenerator(config)

        fp = gen.generate_single("invalid")

        assert fp.shape == (2048,)
        assert fp.sum() == 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_list(self):
        """Test empty SMILES list returns empty array."""
        config = FingerprintConfig(type="morgan")
        gen = FingerprintGenerator(config)

        fps = gen.generate([])

        assert fps.shape == (0, 2048)

    def test_single_molecule(self, valid_smiles: list[str]):
        """Test single molecule returns correct shape."""
        config = FingerprintConfig(type="morgan")
        gen = FingerprintGenerator(config)

        fps = gen.generate([valid_smiles[0]])

        assert fps.shape == (1, 2048)

    def test_generate_single_method(self, valid_smiles: list[str]):
        """Test generate_single returns 1D array."""
        config = FingerprintConfig(type="morgan")
        gen = FingerprintGenerator(config)

        fp = gen.generate_single(valid_smiles[0])

        assert fp.ndim == 1
        assert fp.shape == (2048,)

    def test_unknown_fingerprint_type_raises(self):
        """Test that unknown fingerprint type raises ValueError."""
        config = FingerprintConfig(type="morgan")
        # Manually override to invalid type
        object.__setattr__(config, "type", "unknown_type")

        with pytest.raises(ValueError, match="Unknown fingerprint type"):
            FingerprintGenerator(config)


# ============================================================================
# Repr and Utility Tests
# ============================================================================


class TestUtilities:
    """Tests for utility methods."""

    def test_repr(self):
        """Test string representation."""
        config = FingerprintConfig(type="morgan")
        config.morgan.n_bits = 1024
        gen = FingerprintGenerator(config)

        repr_str = repr(gen)

        assert "FingerprintGenerator" in repr_str
        assert "morgan" in repr_str
        assert "1024" in repr_str

    def test_fingerprint_consistency(self, valid_smiles: list[str]):
        """Test that same SMILES produces same fingerprint."""
        config = FingerprintConfig(type="morgan")
        gen = FingerprintGenerator(config)

        fp1 = gen.generate([valid_smiles[0]])
        fp2 = gen.generate([valid_smiles[0]])

        assert np.array_equal(fp1, fp2)

    def test_different_molecules_different_fps(self, valid_smiles: list[str]):
        """Test that different molecules produce different fingerprints."""
        config = FingerprintConfig(type="morgan")
        gen = FingerprintGenerator(config)

        fps = gen.generate(valid_smiles[:2])

        # Different molecules should have different fingerprints
        assert not np.array_equal(fps[0], fps[1])
