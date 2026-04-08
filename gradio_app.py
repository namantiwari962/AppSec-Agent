"""
gradio_app.py — AppSec Agent Gradio Dashboard
===================================================
Provides a beautiful, interactive OpenEnv-style web interface for the
AppSec Code Reviewer environment on Hugging Face Spaces.
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr

from server.environment import AppSecEnvironment
from models import AppSecAction

# ── Global State (per-session in Gradio) ──────────────────────────────────────
_env: AppSecEnvironment = None
_last_obs = None


def _get_env_info():
    return "AppSec Code Reviewer v1.0 | appsec-openenv-v1"


def handle_reset(difficulty: str):
    global _env, _last_obs
    _env = AppSecEnvironment(task_difficulty=difficulty.lower())
    _last_obs = _env.reset()
    obs_dict = _last_obs.model_dump()
    status = "✅ Environment reset successfully!"
    reward_info = "Reward: N/A | Done: False"
    return (
        status,
        reward_info,
        json.dumps({"observation": obs_dict, "message": "Environment reset."}, indent=2),
    )


def handle_step(action: str):
    global _env, _last_obs
    if _env is None:
        return (
            "⚠️ Please reset the environment first!",
            "",
            json.dumps({"error": "Environment not initialized. Call Reset first."}, indent=2),
        )
    try:
        action_obj = AppSecAction(action=action.lower())
        _last_obs, reward, done, info = _env.step(action_obj)
        obs_dict = _last_obs.model_dump()
        status = f"✅ Step complete! {'🏁 Episode Done!' if done else ''}"
        reward_info = f"Reward: {reward.value:.4f} | Done: {done}"
        response = {
            "observation": obs_dict,
            "reward": reward.value,
            "done": done,
        }
        return status, reward_info, json.dumps(response, indent=2)
    except Exception as e:
        return "❌ Error during step", "", json.dumps({"error": str(e)}, indent=2)


def handle_get_state():
    global _env, _last_obs
    if _env is None or _last_obs is None:
        return (
            "⚠️ Please reset the environment first!",
            "",
            json.dumps({"error": "No active environment session."}, indent=2),
        )
    obs_dict = _last_obs.model_dump()
    status = "✅ State retrieved!"
    reward_info = f"Step count: {_last_obs.step_count}"
    return status, reward_info, json.dumps({"observation": obs_dict}, indent=2)


# ── UI Layout ────────────────────────────────────────────────────────────────

DESCRIPTION_MD = """
# 🛡️ AppSec Agent — AI Security Code Reviewer

An **OpenEnv-compatible** Reinforcement Learning environment for Application Security.
The agent reviews code snippets and decides the optimal security action:
`ignore` • `flag` • `fix` • `escalate`

---
### 🚀 Connect to this environment
```python
from openenv.core import MyEnvAction, MyEnvObservation
with MyEnvClient(base_url="https://namantiwari-appsec-agent.hf.space") as env:
    result = await env.reset()
```

Or connect directly to a running server:
```python
env = MyEnvClient(base_url="https://namantiwari-appsec-agent.hf.space")
```

---
"""

CONTRIBUTE_MD = """
### 🤝 Contribute to this environment
Submit improvements via pull request on Hugging Face Hub:
```bash
openenv fork hf-username/ht-repo-name --hf
```
Then make your changes and submit a pull request:
```bash
cd forked-repo
openenv push hf-username/ht-repo-name --create-pr
```
For more information, see the [OpenEnv documentation](https://github.com/openenv/openenv).
"""

README_MD = """
### 📖 README

**AppSec Agent** is a production-grade RL environment simulating a real-world Application Security (AppSec) code review pipeline.

**Task Difficulties:**
- 🟢 **Easy** — 3 scenarios, common vulnerability patterns
- 🟡 **Medium** — 5 scenarios, mixed severities including false positives
- 🔴 **Hard** — 7 scenarios, complex real-world edge cases

**Actions & Rewards:**
| Action | Best For |
|--------|----------|
| `ignore` | False positives |
| `flag` | Medium/Low real issues |
| `fix` | High severity with known patch |
| `escalate` | Critical severity issues |

**Scoring:** Rewards are deterministic and range from 0.0 to 1.0 per step.
A final score ≥ 0.60 is considered a success.
"""

with gr.Blocks(
    title="AppSec Agent — AI Security Code Reviewer",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    css="""
    .container { max-width: 1100px; margin: auto; }
    .json-output { font-family: 'Courier New', monospace; font-size: 13px; }
    .reward-badge { font-weight: bold; color: #22c55e; font-size: 15px; }
    footer { display: none !important; }
    """,
) as demo:
    with gr.Row():
        gr.Markdown(DESCRIPTION_MD)

    with gr.Row():
        # Left panel — Quick Start
        with gr.Column(scale=1):
            gr.Markdown("### ⚡ Quick Start")
            difficulty_dd = gr.Dropdown(
                choices=["Easy", "Medium", "Hard"],
                value="Easy",
                label="Task Difficulty",
                interactive=True,
            )
            action_dd = gr.Dropdown(
                choices=["ignore", "flag", "fix", "escalate"],
                value="flag",
                label="Action",
                interactive=True,
            )

            with gr.Row():
                step_btn = gr.Button("▶ Step", variant="primary", size="lg")
                reset_btn = gr.Button("🔄 Reset", variant="secondary", size="lg")
                state_btn = gr.Button("📊 Get State", size="lg")

            reward_display = gr.Markdown(
                value="**Reward:** N/A | **Done:** False",
                elem_classes=["reward-badge"],
            )
            status_display = gr.Textbox(
                label="Status",
                value="Reset the environment to begin.",
                interactive=False,
                lines=2,
            )

        # Right panel — JSON Response
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Raw JSON Response")
            json_output = gr.Code(
                label="",
                language="json",
                value='{\n  "message": "Reset the environment to begin."\n}',
                lines=20,
                elem_classes=["json-output"],
            )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown(CONTRIBUTE_MD)
        with gr.Column():
            gr.Markdown(README_MD)

    # ── Event bindings ─────────────────────────────────────────────────────────
    reset_btn.click(
        fn=handle_reset,
        inputs=[difficulty_dd],
        outputs=[status_display, reward_display, json_output],
    )
    step_btn.click(
        fn=handle_step,
        inputs=[action_dd],
        outputs=[status_display, reward_display, json_output],
    )
    state_btn.click(
        fn=handle_get_state,
        inputs=[],
        outputs=[status_display, reward_display, json_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
