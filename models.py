"""
Pydantic models for the AI Security Code Reviewer (AppSec) RL Environment.

Observation, Action, and Reward models as required by the OpenEnv spec.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


# ── Core Models (strictly as specified) ────────────────────────────────────────

class AppSecObservation(BaseModel):
    """State observation presented to the agent at each review step."""
    code_snippet: str = Field(..., description="The code snippet under review.")
    detected_issue: str = Field(..., description="Vulnerability/issue detected by static analysis.")
    severity: str = Field(..., description="Severity level: low | medium | high | critical")
    context: str = Field(..., description="Contextual information: exposure, endpoint, data sensitivity.")
    step_count: int = Field(..., description="Current step index within the episode.")


class AppSecAction(BaseModel):
    """The decision taken by the agent for the given code snippet."""
    action: str = Field(
        ...,
        description="One of: ignore | flag | fix | escalate",
    )


class AppSecReward(BaseModel):
    """Reward signal with human-readable reasoning for transparency."""
    value: float = Field(..., description="Reward value normalized to [0.01, 0.99].")
    reasoning: str = Field(..., description="Explanation of why this reward was assigned.")


# ── Extended Observation for OpenEnv HTTP Transport ────────────────────────────
# The OpenEnv HTTP server serializes step() output. We embed reward metadata
# into an extended observation so the client can reconstruct the full step result.

class AppSecObservationExtended(AppSecObservation):
    """Superset of AppSecObservation used for HTTP transport via OpenEnv server."""
    reward: Optional[float] = Field(None, description="Step reward value.")
    reward_reasoning: Optional[str] = Field(None, description="Reward explanation.")
    done: Optional[bool] = Field(None, description="Whether the episode has ended.")
    info: Optional[Dict[str, Any]] = Field(None, description="Diagnostic metadata dict.")


# ── OpenEnv-required aliases ────────────────────────────────────────────────────
# create_app() requires MyAction and MyObservation to be importable from models.
MyAction = AppSecAction
MyObservation = AppSecObservationExtended