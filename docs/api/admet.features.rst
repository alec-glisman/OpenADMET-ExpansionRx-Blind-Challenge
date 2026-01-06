Features (``admet.features``)
=============================

The ``admet.features`` package provides molecular fingerprint and descriptor
generation for classical machine learning models. It supports multiple
fingerprint types including Morgan, RDKit, MACCS, and Mordred descriptors.

.. automodule:: admet.features
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

FingerprintGenerator
--------------------

Main class for generating molecular fingerprints and descriptors from SMILES.

.. autoclass:: admet.features.FingerprintGenerator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

**Supported Fingerprint Types:**

- **morgan**: Circular fingerprints (default: radius=2, n_bits=2048)
- **rdkit**: RDKit path-based fingerprints
- **maccs**: Fixed 167-bit MACCS structural keys
- **mordred**: Molecular descriptors (~1800 descriptors)

**Example Usage:**

.. code-block:: python

   from admet.features import FingerprintGenerator
   from admet.model.config import FingerprintConfig

   # Generate Morgan fingerprints
   config = FingerprintConfig(type="morgan", radius=2, n_bits=2048)
   generator = FingerprintGenerator(config)

   smiles = ["CCO", "CCCO", "c1ccccc1"]
   features = generator.generate(smiles)
   print(features.shape)  # (3, 2048)

   # Generate MACCS keys
   config_maccs = FingerprintConfig(type="maccs")
   gen_maccs = FingerprintGenerator(config_maccs)
   maccs_features = gen_maccs.generate(smiles)
   print(maccs_features.shape)  # (3, 167)

   # Generate Mordred descriptors
   config_mordred = FingerprintConfig(type="mordred")
   gen_mordred = FingerprintGenerator(config_mordred)
   descriptors = gen_mordred.generate(smiles)
   print(descriptors.shape)  # (3, ~1800)

**Key Methods:**

- ``generate(smiles_list)``: Generate fingerprints for a list of SMILES strings
- ``fingerprint_dim``: Property returning the dimensionality of the fingerprint

**Configuration:**

See :doc:`../guide/configuration` for detailed configuration options via
``FingerprintConfig`` dataclass.
