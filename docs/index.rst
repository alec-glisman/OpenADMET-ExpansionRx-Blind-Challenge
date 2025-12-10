.. image:: _static/images/logo.svg
   :alt: OpenADMET challenge logo
   :align: center
   :width: 360px

OpenADMET Challenge Docs
#########################

Welcome to the OpenADMET + ExpansionRx Blind Challenge docs. This
site contains developer guides, CLI examples, API references, and
visualization tooling to help you build and evaluate ADMET models.

Quick links
-----------

.. panels::

    Quick Start
    ^^^^^^^^^^^

    Create a Python virtualenv, install the package with docs extras,
    and build the docs locally.

    .. code-block:: bash

        # Create virtual environment
        uv venv
        source .venv/bin/activate
        uv pip install -e '.[docs,dev]'

        # Build the documentation
        make -C docs html

        # Run unit tests
        pytest -q

    ---

    API Reference
    ^^^^^^^^^^^^^

    Explore the public package API and modules with detailed docstrings.

    :badge:`Browse API,badge-primary`

    .. link-button:: api/admet.html
        :text: ADMET Package CLI
        :classes: stretched-link

    :badge:`Examples,badge-primary`

    .. link-button:: cli.html
        :text: ADMET Package CLI Examples
        :classes: stretched-link

    ---

    Contributing
    ^^^^^^^^^^^^

    Learn how to run tests and prepare PRs in :doc:`/guide/development`.

    :badge:`Repository,badge-primary`

    .. link-button:: https://github.com/alec-glisman/OpenADMET-ExpansionRx-Blind-Challenge
        :text: Code Repository
        :classes: stretched-link

Documentation sections
----------------------

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/admet
   guide/cli


.. toctree::
   :maxdepth: 1
   :caption: Guides

   guide/overview
   guide/development
   guide/architecture
   guide/data_sources
   guide/splitting
   guide/configuration
   guide/config_reference
   guide/modeling
   guide/hpo
   guide/curriculum
   guide/task_affinity


To build the docs locally, follow one of these methods:
=======================================================

.. tabs::

   .. tab:: Makefile

      .. code-block:: bash

         make -C docs clean && make -C docs html

   .. tab:: Sphinx-Autobuild

      .. code-block:: bash

         sphinx-autobuild docs docs/_build/html --open-browser

   .. tab:: Python

      .. code-block:: bash

         python -m sphinx -b html docs docs/_build/html


Would you like to contribute?
------------------------------

We welcome bug reports and improvements. Please follow the
contribution guidelines in :doc:`/guide/development` and open an issue
for missing documentation, broken examples, or build errors.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
