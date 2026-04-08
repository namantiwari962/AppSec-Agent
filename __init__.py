# Root package marker for openev-project
#
# Ensures the project root is always on sys.path for reliable imports.

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
