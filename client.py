"""
AppSec Code Reviewer — OpenEnv Client

Provides a typed Python client for interacting with the AppSec environment
server over HTTP / WebSocket using the OpenEnv EnvClient protocol.
"""

from typing import Any, Dict
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import AppSecAction, AppSecObservationExtended


class AppSecEnv(EnvClient[AppSecAction, AppSecObservationExtended, State]):
    """
    Client for the AI Security Code Reviewer RL Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with AppSecEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.detected_issue)
        ...     action = AppSecAction(action="fix")
        ...     result = client.step(action)
        ...     print(result.reward)

    Example with Docker:
        >>> client = AppSecEnv.from_docker_image("appsec-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(AppSecAction(action="escalate"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: AppSecAction) -> Dict[str, Any]:
        """Convert AppSecAction to JSON payload for the step WebSocket message."""
        return {"action": action.action}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AppSecObservationExtended]:
        """Parse server HTTP/WS response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        observation = AppSecObservationExtended(
            code_snippet=obs_data.get("code_snippet", ""),
            detected_issue=obs_data.get("detected_issue", ""),
            severity=obs_data.get("severity", "low"),
            context=obs_data.get("context", ""),
            step_count=obs_data.get("step_count", 0),
            reward=obs_data.get("reward"),
            reward_reasoning=obs_data.get("reward_reasoning"),
            done=payload.get("done", False),
            info=obs_data.get("info"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server response into a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# Backward-compatible alias
MyEnv = AppSecEnv
