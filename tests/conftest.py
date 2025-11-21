import sys
from pathlib import Path

# Ensure src path is importable without editable install
root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
