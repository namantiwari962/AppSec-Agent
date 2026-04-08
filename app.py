"""
app.py — AppSec Agent: Unified FastAPI + Gradio Entry Point
============================================================
Runs the OpenEnv-compliant FastAPI backend (for /reset, /step, /state)
AND the Gradio dashboard on the SAME port (7860) using gr.mount_gradio_app.

Endpoints:
  GET  /           → redirects to /ui
  POST /reset      → OpenEnv reset
  POST /step       → OpenEnv step
  GET  /state      → OpenEnv state
  GET  /ui         → Gradio dashboard (with live Security Audit runner)

HuggingFace Spaces requirement: bind to 0.0.0.0:7860.
"""

import io
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Generator, Optional

# ── sys.path bootstrap ─────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── FastAPI / OpenEnv ──────────────────────────────────────────────────────────
try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from exc

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, HTMLResponse

from models import MyAction, MyObservation
from server.environment import MyEnvironment

# Build the FastAPI app with OpenEnv routes
app: FastAPI = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="appsec_code_reviewer",
    max_concurrent_envs=4,
)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root → Gradio UI."""
    return RedirectResponse(url="/ui")


# ── Gradio Dashboard ───────────────────────────────────────────────────────────
import json
import gradio as gr
from server.environment import AppSecEnvironment
from models import AppSecAction


# ── Global env state (per-process; Gradio is single-process by default) ────────
_env: Optional[AppSecEnvironment] = None
_last_obs = None


# ── Environment handlers ───────────────────────────────────────────────────────

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
            "info": info,
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


# ── Security Audit (live streaming logs from inference.py logic) ──────────────

def run_security_audit(difficulty: str, hf_token: str) -> Generator[str, None, None]:
    """
    Runs the full inference.py agent loop in a background thread and streams
    logs live into the Gradio Textbox using a queue + generator pattern.
    """
    import os as _os
    import sys as _sys
    import io as _io
    import queue as _queue
    import threading as _threading
    from typing import List, Optional

    log_queue: _queue.Queue = _queue.Queue()
    accumulated = []

    # Patch print so the inference loop writes into our queue
    class QueueWriter(_io.TextIOBase):
        def write(self, text: str) -> int:
            if text and text != "\n":
                log_queue.put(text)
            return len(text)
        def flush(self):
            pass

    def _run():
        """Execute the inference loop in a sandboxed way."""
        token = hf_token.strip() if hf_token.strip() else _os.environ.get("HF_TOKEN", "")
        if not token:
            log_queue.put("❌ ERROR: HF_TOKEN is required. Set it in the Secret field above.\n")
            log_queue.put("__DONE__")
            return

        from openai import OpenAI
        from server.environment import AppSecEnvironment
        from models import AppSecAction

        API_BASE_URL  = _os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        MODEL_NAME    = _os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

        SYSTEM_PROMPT = """\
You are an expert Application Security (AppSec) engineer performing a code review.
You will be given a code snippet along with a detected vulnerability, its severity, and context.
Your task is to choose EXACTLY ONE action:
  - ignore   : The finding is a false positive. No action needed.
  - flag     : Real issue, but low-medium severity. Needs developer attention.
  - fix      : Real issue with a known safe patch. Can be auto-remediated.
  - escalate : Critical or complex issue requiring an immediate human security expert.
Respond in this EXACT format (no extra text):
ACTION: <ignore|flag|fix|escalate>
REASON: <one concise sentence explaining your decision>
"""

        VALID_ACTIONS = {"ignore", "flag", "fix", "escalate"}

        def _log(msg: str):
            log_queue.put(msg + "\n")

        def _parse_action(llm_response: str) -> str:
            for line in llm_response.splitlines():
                line = line.strip()
                if line.upper().startswith("ACTION:"):
                    candidate = line.split(":", 1)[1].strip().lower()
                    if candidate in VALID_ACTIONS:
                        return candidate
            lower = llm_response.lower()
            for act in ("escalate", "fix", "flag", "ignore"):
                if act in lower:
                    return act
            return "flag"

        def _fallback(obs) -> str:
            try:
                sev = obs.severity.lower()
                if sev == "critical": return "escalate"
                elif sev == "high":   return "fix"
                elif sev == "medium": return "flag"
                else:                 return "ignore"
            except AttributeError:
                return "flag"

        client = OpenAI(base_url=API_BASE_URL, api_key=token)
        tasks = [difficulty.lower()] if difficulty.lower() != "all" else ["easy", "medium", "hard"]

        for diff in tasks:
            task_name = f"appsec-code-review-{diff}"
            env = AppSecEnvironment(task_difficulty=diff)

            _log(f"[START]")
            _log(f"# task={task_name} env=appsec-openenv-v1 model={MODEL_NAME}")

            obs = env.reset()
            all_rewards: List[float] = []
            steps_taken: int = 0
            done: bool = False

            while not done:
                steps_taken += 1
                error_msg: Optional[str] = None
                action_str = "flag"

                try:
                    prompt = (
                        f"=== CODE REVIEW REQUEST (Step {obs.step_count + 1}) ===\n\n"
                        f"SEVERITY : {obs.severity.upper()}\n"
                        f"DETECTED : {obs.detected_issue}\n"
                        f"CONTEXT  : {obs.context}\n\n"
                        f"CODE SNIPPET:\n```\n{obs.code_snippet}\n```\n\n"
                        f"What is your security decision?"
                    )
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=150,
                    )
                    raw = response.choices[0].message.content or ""
                    action_str = _parse_action(raw)
                except Exception as exc:
                    error_msg = f"LLM Error: {str(exc)[:80]}".replace("\n", " ")
                    action_str = _fallback(obs)

                try:
                    action_obj = AppSecAction(action=action_str)
                    obs, reward, done, info = env.step(action_obj)
                    all_rewards.append(reward.value)
                    _log(
                        f"[STEP] step={steps_taken} action={action_str} "
                        f"reward={reward.value:.4f} done={str(done).lower()} "
                        f"error={error_msg or 'null'}"
                    )
                except Exception as env_exc:
                    error_msg = f"Env Error: {str(env_exc)[:80]}".replace("\n", " ")
                    _log(
                        f"[STEP] step={steps_taken} action=flag reward=0.0000 "
                        f"done=true error={error_msg}"
                    )
                    all_rewards.append(0.0)
                    break

            final_score = env.compute_final_score()
            success = final_score >= 0.60
            rewards_str = ",".join(f"{r:.4f}" for r in all_rewards)
            _log(
                f"[END] success={str(success).lower()} steps={steps_taken} "
                f"score={final_score:.4f} rewards={rewards_str}"
            )
            _log("")

        log_queue.put("__DONE__")

    thread = _threading.Thread(target=_run, daemon=True)
    thread.start()

    buffer = "🚀 Starting Security Audit...\n\n"
    yield buffer

    while True:
        try:
            line = log_queue.get(timeout=60)
            if line == "__DONE__":
                break
            accumulated.append(line)
            buffer = "🚀 Security Audit Running...\n\n" + "".join(accumulated)
            yield buffer
        except _queue.Empty:
            break

    final = "✅ Audit Complete!\n\n" + "".join(accumulated)
    yield final


# ── Gradio Blocks UI ───────────────────────────────────────────────────────────

DESCRIPTION_MD = """
# 🛡️ AppSec Agent — AI Security Code Reviewer

An **OpenEnv-compatible** Reinforcement Learning environment for Application Security.
The agent reviews code snippets and decides the optimal security action:
`ignore` • `flag` • `fix` • `escalate`

---
### 🔗 OpenEnv API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST   | Start a new episode |
| `/step`  | POST   | Submit an agent action |
| `/state` | GET    | Get current episode state |

---
"""

README_MD = """
### 📖 README

**Task Difficulties:**
- 🟢 **Easy** — 3 scenarios, common patterns
- 🟡 **Medium** — 5 scenarios, mixed severities & false positives
- 🔴 **Hard** — 7 scenarios, complex real-world edge cases

**Actions & Rewards:**
| Action | Best For |
|--------|----------|
| `ignore`   | False positives |
| `flag`     | Medium/Low real issues |
| `fix`      | High severity with known patch |
| `escalate` | Critical severity issues |

**Scoring:** Rewards range 0.0 – 1.0. Final score ≥ 0.60 is a success.
"""

with gr.Blocks(
    title="AppSec Agent — AI Security Code Reviewer",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    css="""
    .container { max-width: 1200px; margin: auto; }
    .json-output { font-family: 'Courier New', monospace; font-size: 13px; }
    .reward-badge { font-weight: bold; color: #22c55e; font-size: 15px; }
    .audit-log { font-family: 'Courier New', monospace; font-size: 12px; background: #0d1117; color: #58a6ff; }
    footer { display: none !important; }
    """,
) as demo:

    gr.Markdown(DESCRIPTION_MD)

    # ── Tab 1: Manual Environment Control ─────────────────────────────────────
    with gr.Tab("⚡ Manual Control"):
        with gr.Row():
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
                    step_btn  = gr.Button("▶ Step",      variant="primary",   size="lg")
                    reset_btn = gr.Button("🔄 Reset",     variant="secondary", size="lg")
                    state_btn = gr.Button("📊 Get State",                       size="lg")

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

            with gr.Column(scale=1):
                gr.Markdown("### 📋 Raw JSON Response")
                json_output = gr.Code(
                    label="",
                    language="json",
                    value='{\n  "message": "Reset the environment to begin."\n}',
                    lines=20,
                    elem_classes=["json-output"],
                )

        reset_btn.click(fn=handle_reset,    inputs=[difficulty_dd],  outputs=[status_display, reward_display, json_output])
        step_btn.click( fn=handle_step,     inputs=[action_dd],      outputs=[status_display, reward_display, json_output])
        state_btn.click(fn=handle_get_state, inputs=[],              outputs=[status_display, reward_display, json_output])

    # ── Tab 2: Run Security Audit (LLM Agent) ─────────────────────────────────
    with gr.Tab("🔍 Run Security Audit"):
        gr.Markdown("""
### 🔍 Run Full LLM Security Audit

Runs the **AI agent** (powered by HuggingFace Inference Router) through all
difficulty levels and streams live output in the [START]/[STEP]/[END] OpenEnv format.

> **Note:** You must provide your `HF_TOKEN` below. This is never stored.
""")
        with gr.Row():
            with gr.Column(scale=1):
                audit_difficulty = gr.Dropdown(
                    choices=["easy", "medium", "hard", "all"],
                    value="easy",
                    label="Difficulty to Run",
                    interactive=True,
                )
                audit_token = gr.Textbox(
                    label="🔑 HF_TOKEN (required for LLM calls)",
                    placeholder="hf_...",
                    type="password",
                    lines=1,
                )
                audit_btn = gr.Button(
                    "🚀 Run Security Audit",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=2):
                audit_log = gr.Textbox(
                    label="📡 Live Audit Logs",
                    value="Click 'Run Security Audit' to begin...",
                    lines=28,
                    max_lines=60,
                    interactive=False,
                    elem_classes=["audit-log"],
                )

        audit_btn.click(
            fn=run_security_audit,
            inputs=[audit_difficulty, audit_token],
            outputs=[audit_log],
        )

    # ── Tab 3: About ───────────────────────────────────────────────────────────
    with gr.Tab("📖 About"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
### 🤝 Contribute to this environment
```bash
openenv fork namantiwari/appsec-agent --hf
cd forked-repo
openenv push namantiwari/appsec-agent --create-pr
```
For more information, see the [OpenEnv documentation](https://github.com/openenv/openenv).
""")
            with gr.Column():
                gr.Markdown(README_MD)


# ── Mount Gradio onto FastAPI ──────────────────────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/ui")


# ── Entry point ────────────────────────────────────────────────────────────────
def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
