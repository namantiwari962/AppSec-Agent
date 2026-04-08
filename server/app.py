"""
FastAPI application for the AppSec Code Reviewer RL Environment.
"""

import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path (belt-and-suspenders alongside __init__.py)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError("openenv-core is required. Install with: pip install openenv-core") from exc

from fastapi.responses import RedirectResponse
from models import MyAction, MyObservation
from server.environment import MyEnvironment

app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="appsec_code_reviewer",
    max_concurrent_envs=4,
)

@app.get("/")
async def root_redirect():
    """Redirect root to the web UI."""
    return RedirectResponse(url="/web/")

def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AppSec Code Reviewer Server")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
