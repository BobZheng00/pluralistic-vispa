"""
Adds the gitignored third_party/ checkouts (populated by
scripts/setup_dependencies.*) to sys.path so bridge modules in this package
can `import src...` / `import repe` directly from them, without vendoring
their code. See steering/README.md for why.
"""

import sys
from pathlib import Path

THIRD_PARTY_DIR = Path(__file__).resolve().parents[1] / "third_party"


def add_to_path(*names: str) -> None:
    for name in names:
        path = THIRD_PARTY_DIR / name
        if not path.is_dir():
            raise FileNotFoundError(
                f"third_party/{name} not found. Run scripts/setup_dependencies.sh "
                f"(or .ps1 on Windows) first."
            )
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
