import sys
from pathlib import Path

# Make the URDFParser package importable as ``URDFParser`` (the package
# directory's PARENT must be on sys.path). These tests live inside the
# submodule so they can run standalone (CI on the URDFParser repo) as well
# as from the GRiD super-project.
_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))
