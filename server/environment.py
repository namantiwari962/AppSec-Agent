"""
AI Security Code Reviewer — AppSec Agent RL Environment.

Simulates a real-world Application Security review pipeline where an agent
must classify code vulnerabilities and decide the appropriate response:
  - ignore   : false positive, no action needed
  - flag     : real issue, needs developer attention
  - fix      : real issue, can be auto-remediated
  - escalate : critical issue, requires human security expert

step() returns: Tuple[AppSecObservation, AppSecReward, bool, Dict]
"""

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path so `from models import ...` resolves
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    AppSecAction,
    AppSecObservation,
    AppSecObservationExtended,
    AppSecReward,
)


# ── Deterministic Reward Table ─────────────────────────────────────────────────
# Keys  : (severity: str, is_false_positive: bool, action: str)
# Values: (reward: float [0.0–1.0], reasoning: str)

REWARD_TABLE: Dict[Tuple[str, bool, str], Tuple[float, str]] = {
    # ── CRITICAL — Real vulnerability ─────────────────────────────────────────
    ("critical", False, "escalate"): (
        1.00, "Optimal: Critical vulnerability correctly escalated to human security team.",
    ),
    ("critical", False, "fix"): (
        0.70, "Partial: Auto-fix applied, but critical issues demand human oversight — escalate first.",
    ),
    ("critical", False, "flag"): (
        0.20, "Insufficient: Critical vulnerability merely flagged; must be escalated immediately.",
    ),
    ("critical", False, "ignore"): (
        0.00, "CRITICAL FAILURE: Ignored a critical vulnerability — catastrophic breach risk.",
    ),
    # ── CRITICAL — False positive ─────────────────────────────────────────────
    ("critical", True, "ignore"): (
        1.00, "Optimal: Correctly identified false positive, avoided wasteful escalation.",
    ),
    ("critical", True, "flag"): (
        0.30, "Over-reaction: Flagged a false positive — unnecessary developer distraction.",
    ),
    ("critical", True, "fix"): (
        0.10, "Resource waste: Applied fix to a non-existent vulnerability.",
    ),
    ("critical", True, "escalate"): (
        0.00, "Worst outcome: Escalated a false positive — maximum disruption to security team.",
    ),
    # ── HIGH — Real vulnerability ──────────────────────────────────────────────
    ("high", False, "fix"): (
        1.00, "Optimal: High-severity vulnerability efficiently auto-remediated.",
    ),
    ("high", False, "escalate"): (
        0.65, "Acceptable: Escalated instead of fixing — valid but slightly less efficient.",
    ),
    ("high", False, "flag"): (
        0.40, "Insufficient: Flagged but not resolved; high-severity requires fix or escalation.",
    ),
    ("high", False, "ignore"): (
        0.00, "FAILURE: High-severity vulnerability ignored — serious exploitation risk.",
    ),
    # ── HIGH — False positive ─────────────────────────────────────────────────
    ("high", True, "ignore"): (
        1.00, "Optimal: False positive correctly dismissed — no wasted cycles.",
    ),
    ("high", True, "flag"): (
        0.40, "Minor noise: Unnecessary flag for a non-issue — some developer time lost.",
    ),
    ("high", True, "fix"): (
        0.15, "Resource waste: Fix applied to non-existent high-severity issue.",
    ),
    ("high", True, "escalate"): (
        0.00, "Escalation abuse: Human reviewer time wasted on a false positive.",
    ),
    # ── MEDIUM — Real vulnerability ────────────────────────────────────────────
    ("medium", False, "flag"): (
        1.00, "Optimal: Medium-severity issue correctly flagged for developer review.",
    ),
    ("medium", False, "fix"): (
        0.75, "Good: Proactive fix applied; flag would be sufficient but fix is acceptable.",
    ),
    ("medium", False, "escalate"): (
        0.25, "Over-escalation: Medium issues rarely require immediate human intervention.",
    ),
    ("medium", False, "ignore"): (
        0.10, "Risk: Medium vulnerability ignored — potential exploitation pathway left open.",
    ),
    # ── MEDIUM — False positive ────────────────────────────────────────────────
    ("medium", True, "ignore"): (
        1.00, "Optimal: Correctly identified false positive — no alert fatigue introduced.",
    ),
    ("medium", True, "flag"): (
        0.50, "Mild noise: Medium false positive flagged — minor developer distraction.",
    ),
    ("medium", True, "fix"): (
        0.20, "Unnecessary: Fix resources spent on non-existent medium vulnerability.",
    ),
    ("medium", True, "escalate"): (
        0.05, "Over-reaction: Escalated a false positive at medium severity.",
    ),
    # ── LOW — Real vulnerability ───────────────────────────────────────────────
    ("low", False, "flag"): (
        0.85, "Good: Low-severity issue logged and tracked appropriately.",
    ),
    ("low", False, "ignore"): (
        0.65, "Acceptable: Low-severity noise suppressed; minor risk acknowledged.",
    ),
    ("low", False, "fix"): (
        0.50, "Overkill but valid: Fixed a low-severity issue — not cost-efficient.",
    ),
    ("low", False, "escalate"): (
        0.10, "Over-escalation: Low-severity issue does not warrant human review.",
    ),
    # ── LOW — False positive ───────────────────────────────────────────────────
    ("low", True, "ignore"): (
        1.00, "Optimal: Low-severity false positive correctly dismissed.",
    ),
    ("low", True, "flag"): (
        0.60, "Minor noise: Low-severity false positive flagged — negligible impact.",
    ),
    ("low", True, "fix"): (
        0.30, "Unnecessary: Fixed a non-existent low-severity issue.",
    ),
    ("low", True, "escalate"): (
        0.10, "Over-escalation: Escalated a low-severity false positive.",
    ),
}

MAX_STEPS: Dict[str, int] = {"easy": 3, "medium": 5, "hard": 7}

FALLBACK_SCENARIOS: List[Dict] = [
    {
        "id": "fallback_sqli",
        "code_snippet": 'query = f"SELECT * FROM users WHERE id = \'{user_id}\'"',
        "detected_issue": "SQL Injection — Direct string interpolation in SQL query.",
        "severity": "critical",
        "context": "Public-facing login endpoint. user_id sourced from raw HTTP request params.",
        "is_false_positive": False,
        "optimal_action": "fix",
    }
]


# ═══════════════════════════════════════════════════════════════════════════════
class AppSecEnvironment(Environment):
    """
    Core Application Security Code Review RL Environment.

    step() returns Tuple[AppSecObservation, AppSecReward, bool, Dict].
    This is the canonical interface used by inference.py and unit tests.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_difficulty: str = "easy") -> None:
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self.task_difficulty = task_difficulty.lower()
        self.scenarios: List[Dict] = []
        self.current_idx: int = 0
        self.action_history: List[str] = []
        self.episode_rewards: List[float] = []
        self.max_steps: int = MAX_STEPS.get(self.task_difficulty, 5)
        self._load_task()

    # ── Task loading ────────────────────────────────────────────────────────────

    def _load_task(self) -> None:
        """Load task scenarios from tasks/<difficulty>.json."""
        tasks_dir = Path(__file__).resolve().parent.parent / "tasks"
        task_file = tasks_dir / f"{self.task_difficulty}.json"
        if task_file.exists():
            with open(task_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.scenarios = data.get("scenarios", [])
        else:
            self.scenarios = list(FALLBACK_SCENARIOS)
        # Clamp max_steps to available scenarios
        self.max_steps = min(self.max_steps, len(self.scenarios))

    # ── OpenEnv interface ────────────────────────────────────────────────────────

    def reset(self) -> AppSecObservation:
        """Reset the environment and return the initial observation."""
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self.current_idx = 0
        self.action_history = []
        self.episode_rewards = []
        return self._build_observation()

    def step(
        self, action: AppSecAction
    ) -> Tuple[AppSecObservation, AppSecReward, bool, Dict]:
        """
        Execute one code-review decision.

        Returns:
            obs     : AppSecObservation — next state
            reward  : AppSecReward      — value [0,1] + reasoning string
            done    : bool              — True when episode ends
            info    : Dict              — diagnostic metadata
        """
        self._state.step_count += 1
        scenario = self.scenarios[self.current_idx]

        # ── Compute base reward ──────────────────────────────────────────────
        severity: str = scenario["severity"].lower()
        is_fp: bool = bool(scenario.get("is_false_positive", False))
        act: str = action.action.lower()

        key = (severity, is_fp, act)
        base_reward, reasoning = REWARD_TABLE.get(
            key,
            (0.0, f"Unknown key: severity={severity}, fp={is_fp}, action={act}"),
        )

        # ── Loop-prevention penalty ──────────────────────────────────────────
        self.action_history.append(act)
        loop_penalty = 0.0
        if len(self.action_history) >= 3 and len(set(self.action_history[-3:])) == 1:
            loop_penalty = 0.15
            count = self.action_history.count(act)
            reasoning += (
                f" [Loop penalty -0.15: action '{act}' repeated {count}x in a row.]"
            )

        final_reward = round(max(0.0001, min(0.9999, base_reward - loop_penalty)), 4)
        self.episode_rewards.append(final_reward)

        # ── Advance scenario pointer ─────────────────────────────────────────
        prev_scenario = scenario
        self.current_idx += 1
        done = (
            self.current_idx >= len(self.scenarios)
            or self._state.step_count >= self.max_steps
        )

        # ── Build next observation ───────────────────────────────────────────
        if done:
            mean_r = sum(self.episode_rewards) / len(self.episode_rewards)
            next_obs = AppSecObservation(
                code_snippet="# [SESSION COMPLETE — all scenarios reviewed]",
                detected_issue="No further issues. Episode ended.",
                severity="low",
                context=(
                    f"Episode finished in {self._state.step_count} step(s). "
                    f"Mean reward: {mean_r:.4f}."
                ),
                step_count=self._state.step_count,
            )
        else:
            next_obs = self._build_observation()

        reward_obj = AppSecReward(value=final_reward, reasoning=reasoning)

        info: Dict[str, Any] = {
            "scenario_id": prev_scenario.get("id", "unknown"),
            "optimal_action": prev_scenario.get("optimal_action", "unknown"),
            "is_false_positive": is_fp,
            "severity": severity,
            "action_taken": act,
            "base_reward": base_reward,
            "loop_penalty_applied": loop_penalty > 0,
            "episode_rewards_so_far": list(self.episode_rewards),
            "current_mean_reward": sum(self.episode_rewards) / len(self.episode_rewards),
        }

        return next_obs, reward_obj, done, info

    @property
    def state(self) -> State:
        return self._state

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_observation(self) -> AppSecObservation:
        s = self.scenarios[self.current_idx]
        return AppSecObservation(
            code_snippet=s["code_snippet"],
            detected_issue=s["detected_issue"],
            severity=s["severity"],
            context=s["context"],
            step_count=self._state.step_count,
        )

    def compute_final_score(self) -> float:
        """Return normalized (0.0, 1.0) episode score."""
        if not self.episode_rewards:
            return 0.0001
        raw_score = sum(self.episode_rewards) / len(self.episode_rewards)
        return round(max(0.0001, min(0.9999, raw_score)), 4)


# ═══════════════════════════════════════════════════════════════════════════════
class MyEnvironment(AppSecEnvironment):
    """
    OpenEnv HTTP-transport-compatible wrapper.

    Overrides step() to return AppSecObservationExtended (with embedded reward/done)
    so the OpenEnv FastAPI server can serialize the full step result over HTTP/WS.
    The inference.py script uses AppSecEnvironment directly for the full tuple.
    """

    def __init__(self) -> None:
        difficulty = os.environ.get("TASK_DIFFICULTY", "easy")
        super().__init__(task_difficulty=difficulty)

    def reset(self) -> AppSecObservationExtended:
        obs = super().reset()
        return AppSecObservationExtended(
            **obs.model_dump(),
            reward=0.0,
            reward_reasoning="Environment reset. Initial state.",
            done=False,
            info={},
        )

    def step(self, action: AppSecAction) -> AppSecObservationExtended:  # type: ignore[override]

        # Prevent stepping after episode is done
        if self.current_idx >= len(self.scenarios):
            raise Exception("Episode already finished. Call reset().")

        obs, reward, done, info = super().step(action)

        return AppSecObservationExtended(
            **obs.model_dump(),
            reward=reward.value,
            reward_reasoning=reward.reasoning,
            done=done,
            info=info,
        )
