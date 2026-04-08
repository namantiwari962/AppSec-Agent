# Server package for OpenEnv AI AppSec Code Reviewer
#
# Ensures the project root is on sys.path so that `from models import ...`
# works regardless of how the server is started (uvicorn, uv run, Docker, etc.)

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
