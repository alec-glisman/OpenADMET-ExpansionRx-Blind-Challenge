#!/usr/bin/env bash
set -euo pipefail

# Build docs (do not fail the commit if the build fails — we only want to
# attempt to create or update docs). Any build errors will be printed but
# will not cause a non-zero exit code.
echo "[pre-commit] Building docs (errors will not fail commit)..."
if ! make -C docs html; then
  echo "[pre-commit] docs build failed but commit will continue. See above for details."
fi

# Stage new/untracked files under docs/ (but exclude build artifacts in _build/)
echo "[pre-commit] Staging new documentation files (if any)..."
git ls-files --others --exclude-standard -- docs | grep -v '^docs/_build' | while IFS= read -r f; do
  if [[ -n "$f" ]]; then
    echo "[pre-commit] Staging: $f"
    git add -- "$f"
  fi
done

echo "[pre-commit] Done (docs build attempted, new docs staged)."
exit 0
