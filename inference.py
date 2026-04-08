"""
inference.py — AI Security Code Reviewer (AppSec Agent)

Runs the AppSec RL environment with an LLM-based agent and outputs
structured logs in the exact [START] / [STEP] / [END] OpenEnv format.

Environment variables:
    HF_TOKEN       (required)  — HuggingFace token used as API key
    API_BASE_URL   (optional)  — Inference endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     (optional)  — Model to use (default: Qwen/Qwen2.5-7B-Instruct)
    TASK_DIFFICULTY(optional)  — easy | medium | hard (default: easy)

Output format (exact, no deviation):
    [START]
    [STEP] step=1 action=fix reward=0.85 done=false error=null
    ...
    [END] success=true steps=3 score=0.95 rewards=0.85,1.00,1.00
"""

import os
import sys
from typing import List, Optional
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from openai import OpenAI

from server.environment import AppSecEnvironment
from models import AppSecAction


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
TASK_DIFFICULTY: str = os.environ.get("TASK_DIFFICULTY", "easy")

VALID_ACTIONS = {"ignore", "flag", "fix", "escalate"}

SYSTEM_PROMPT = """\
You are an expert Application Security (AppSec) engineer performing a code review.

You will be given a code snippet along with a detected vulnerability, its severity, and context.
Your task is to choose EXACTLY ONE action:

  - ignore   : The finding is a false positive. No action needed.
  - flag     : Real issue, but low-medium severity. Needs developer attention.
  - fix      : Real issue with a known safe patch. Can be auto-remediated.
  - escalate : Critical or complex issue requiring an immediate human security expert.

Decision heuristics:
  - CRITICAL real issue    → escalate (or fix if a direct patch is obvious and safe)
  - HIGH real issue        → fix (or escalate if architectural change is needed)
  - MEDIUM real issue      → flag (or fix if patch is trivial)
  - LOW real issue         → flag or ignore (weigh noise vs. risk)
  - False positive         → ignore (do NOT waste escalation capacity)

Respond in this EXACT format (no extra text):
ACTION: <ignore|flag|fix|escalate>
REASON: <one concise sentence explaining your decision>
"""


# ── Logging helpers ────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print("[START]", flush=True)
    print(f"# task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM action extraction ──────────────────────────────────────────────────────

def parse_action(llm_response: str) -> str:
    """Extract action from LLM response. Falls back to 'flag' on parse failure."""
    for line in llm_response.splitlines():
        line = line.strip()
        if line.upper().startswith("ACTION:"):
            candidate = line.split(":", 1)[1].strip().lower()
            if candidate in VALID_ACTIONS:
                return candidate
    # Fallback: scan for any valid action keyword
    lower = llm_response.lower()
    for act in ("escalate", "fix", "flag", "ignore"):
        if act in lower:
            return act
    return "flag"  # Safe default — never silently ignore


def call_llm(client: OpenAI, obs_prompt: str) -> str:
    """Call the LLM and return the raw text response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_prompt},
        ],
        temperature=0.1,   # Fixed: Lower temperature for even more stability
        max_tokens=150,    # Fixed: Prevent potential timeout by requesting fewer tokens
    )
    return response.choices[0].message.content or ""


def build_prompt(obs) -> str:
    """Build the user-facing prompt from an AppSecObservation."""
    return (
        f"=== CODE REVIEW REQUEST (Step {obs.step_count + 1}) ===\n\n"
        f"SEVERITY : {obs.severity.upper()}\n"
        f"DETECTED : {obs.detected_issue}\n"
        f"CONTEXT  : {obs.context}\n\n"
        f"CODE SNIPPET:\n```\n{obs.code_snippet}\n```\n\n"
        f"What is your security decision?"
    )


def fallback_action(obs) -> str:
    """Rule-based fallback action if the LLM API fails, preventing a crash."""
    try:
        sev = obs.severity.lower()
        if sev == "critical":
            return "escalate"
        elif sev == "high":
            return "fix"
        elif sev == "medium":
            return "flag"
        else:
            return "ignore"
    except AttributeError:
        return "flag"


# ── Main episode loop ──────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("ERROR: HF_TOKEN environment variable is required.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    # Run all tasks: easy → medium → hard
    tasks = ["easy", "medium", "hard"]

    for difficulty in tasks:

        TASK_NAME = f"appsec-code-review-{difficulty}"

        env = AppSecEnvironment(task_difficulty=difficulty)

        log_start(task=TASK_NAME, env="appsec-openenv-v1", model=MODEL_NAME)

        # ── Reset ──────────────────────────────────────────────────────────────
        obs = env.reset()
        all_rewards: List[float] = []
        steps_taken: int = 0
        done: bool = False

        # ── Episode loop ───────────────────────────────────────────────────────
        while not done:
            steps_taken += 1
            error_msg: Optional[str] = None
            action_str: str = "flag"

            # 1) Try to use LLM for action selection
            try:
                prompt = build_prompt(obs)
                raw_response = call_llm(client, prompt)
                action_str = parse_action(raw_response)
            except Exception as exc:
                # Handle LLM failure gracefully
                error_msg = f"LLM Error: {str(exc)[:80]}".replace("\n", " ")
                action_str = fallback_action(obs)

            # 2) Execute action in Environment
            try:
                action = AppSecAction(action=action_str)
                obs, reward, done, info = env.step(action)

                all_rewards.append(reward.value)

                log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=reward.value,
                    done=done,
                    error=error_msg,
                )

            except Exception as env_exc:
                # Critical environment error
                error_msg = f"Env Error: {str(env_exc)[:80]}".replace("\n", " ")
                log_step(
                    step=steps_taken,
                    action="flag",
                    reward=0.01,
                    done=True,
                    error=error_msg,
                )
                all_rewards.append(0.01)
                break

        # ── Score computation (INSIDE LOOP) ────────────────────────────────────
        final_score: float = env.compute_final_score()
        success: bool = final_score >= 0.60

        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=all_rewards,
        )


if __name__ == "__main__":
    main()