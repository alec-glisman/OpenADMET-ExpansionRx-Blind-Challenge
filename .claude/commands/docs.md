# Build Documentation

Build and preview Sphinx documentation.

## Build HTML Docs

```bash
make -C docs html
```

## Live Preview (Auto-reload)

```bash
sphinx-autobuild docs docs/_build/html
```

Then open http://127.0.0.1:8000 in browser.

## Clean Build

```bash
make -C docs clean html
```

## Documentation Structure

```
docs/
├── conf.py          # Sphinx configuration
├── index.rst        # Main index
├── guide/           # User guides
│   ├── configuration.rst
│   ├── hpo.rst
│   └── ...
├── api/             # API reference
└── _build/html/     # Built HTML output
```

## Style

Uses scienceplots for matplotlib figures.

## Check Links

```bash
make -C docs linkcheck
```
