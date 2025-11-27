# Makefile for building Sphinx documentation

SPHINXBUILD ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html clean live

help:
	@echo "Make targets: html, clean, live"
	@echo "  make html   - build HTML docs into docs/_build/html"
	@echo "  make clean  - remove build artifacts"
	@echo "  make live   - start sphinx-autobuild for live reload (requires sphinx-autobuild)"

html:
	@echo "Building HTML documentation..."
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo "HTML docs built at $(BUILDDIR)/html"

clean:
	@echo "Removing build directory $(BUILDDIR)"
	rm -rf "$(BUILDDIR)"

live:
	@echo "Starting live-reload server (sphinx-autobuild)..."
	$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" --open-browser
