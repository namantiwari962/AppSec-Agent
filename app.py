"""
app.py — AppSec Agent: Unified FastAPI + Premium Gradio UI
============================================================
Restores the original premium look/feel while maintaining the 
unified FastAPI/OpenEnv backend and fixing deployment errors.
"""

import os
import sys
import json
import queue
import threading
from pathlib import Path
from typing import Generator, Optional, List

# ── sys.path bootstrap ─────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── FastAPI / OpenEnv ──────────────────────────────────────────────────────────
try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError("openenv-core is required. Install with: pip install openenv-core") from exc

from fastapi import FastAPI
from models import MyAction, MyObservation
from server.environment import MyEnvironment, AppSecEnvironment
from models import AppSecAction

# Build the FastAPI app with OpenEnv routes (/reset, /step, /state)
app: FastAPI = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="appsec_code_reviewer",
    max_concurrent_envs=4,
)

# ── Gradio Logic ───────────────────────────────────────────────────────────────
import gradio as gr

# Global session state
_env: Optional[AppSecEnvironment] = None
_last_obs = None

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
        json.dumps({"observation": obs_dict, "message": "Resetted."}, indent=2),
    )

def handle_step(action: str):
    global _env, _last_obs
    if _env is None:
        return ("⚠️ Reset first!", "", json.dumps({"error": "No env."}, indent=2))
    try:
        action_obj = AppSecAction(action=action.lower())
        _last_obs, reward, done, info = _env.step(action_obj)
        obs_dict = _last_obs.model_dump()
        status = f"✅ Step complete! {'🏁 Done!' if done else ''}"
        reward_info = f"Reward: {reward.value:.4f} | Done: {done}"
        return status, reward_info, json.dumps({"observation": obs_dict, "reward": reward.value, "done": done, "info": info}, indent=2)
    except Exception as e:
        return "❌ Error", "", json.dumps({"error": str(e)}, indent=2)

def handle_get_state():
    global _env, _last_obs
    if _env is None or _last_obs is None:
        return ("⚠️ Reset first!", "", json.dumps({"error": "No session."}, indent=2))
    return ("✅ State retrieved!", f"Steps: {_last_obs.step_count}", json.dumps({"observation": _last_obs.model_dump()}, indent=2))

def run_security_audit(difficulty: str, hf_token: str) -> Generator[str, None, None]:
    from openai import OpenAI
    log_queue = queue.Queue()
    accumulated = []
    
    def _run():
        token = hf_token.strip() or os.environ.get("HF_TOKEN", "")
        if not token:
            log_queue.put("❌ ERROR: HF_TOKEN required.\n")
            log_queue.put("__DONE__")
            return
        
        client = OpenAI(base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"), api_key=token)
        tasks = [difficulty.lower()] if difficulty.lower() != "all" else ["easy", "medium", "hard"]
        
        for diff in tasks:
            env = AppSecEnvironment(task_difficulty=diff)
            log_queue.put(f"[START]\n# task=appsec-{diff} env=appsec-v1\n")
            obs = env.reset()
            done = False
            step = 0
            while not done:
                step += 1
                try:
                    res = client.chat.completions.create(
                        model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
                        messages=[
                            {"role":"system","content":"Respond with ACTION: <ignore|flag|fix|escalate>\nREASON: <one sentence>"},
                            {"role":"user","content":str(obs)}
                        ],
                        temperature=0.1, max_tokens=150
                    )
                    act_str = "flag"
                    content = res.choices[0].message.content or ""
                    for line in content.splitlines():
                        if "ACTION:" in line.upper(): act_str = line.split(":")[-1].strip().lower()
                    
                    obs, reward, done, _ = env.step(AppSecAction(action=act_str))
                    log_queue.put(f"[STEP] step={step} action={act_str} reward={reward.value:.4f} done={str(done).lower()}\n")
                except Exception as e:
                    log_queue.put(f"ERROR: {str(e)}\n")
                    break
            log_queue.put(f"[END] score={env.compute_final_score():.4f}\n\n")
        log_queue.put("__DONE__")

    threading.Thread(target=_run, daemon=True).start()
    yield "🚀 Starting Security Audit...\n"
    while True:
        line = log_queue.get()
        if line == "__DONE__": break
        accumulated.append(line)
        yield "".join(accumulated)

# ── UI Markdown & Aesthetics ────────────────────────────────────────────────────

DESCRIPTION_MD = """
# 🛡️ AppSec Agent — AI Security Code Reviewer
An **OpenEnv-compatible** Reinforcement Learning environment for Application Security.
The agent reviews code snippets and decides the optimal security action:
`ignore` • `flag` • `fix` • `escalate`
---
"""

README_MD = """
### 📖 README
**AppSec Agent** simulates a real-world Application Security review pipeline.
**Task Difficulties:**
- 🟢 **Easy** — 3 scenarios, common patterns
- 🟡 **Medium** — 5 scenarios, false positives included
- 🔴 **Hard** — 7 scenarios, complex real-world edge cases
**Actions & Rewards:** Rewards (0.0–1.0) reward efficiency and accuracy.
"""

CONTRIBUTE_MD = """
### 🤝 Contribute
Submit improvements via pull request on Hugging Face Hub:
```bash
openenv fork namantiwari/appsec-agent --hf
```
"""

# ── Gradio Layout ─────────────────────────────────────────────────────────────

with gr.Blocks(
    title="AppSec Agent — AI Security Code Reviewer",
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"),
    css="""
    .container { max-width: 1100px; margin: auto; }
    .json-output { font-family: 'Courier New', monospace; font-size: 13px; }
    .reward-badge { font-weight: bold; color: #22c55e; font-size: 15px; }
    .audit-terminal { font-family: 'Courier New', monospace; background: #0d1117; color: #58a6ff; }
    footer { display: none !important; }
    """,
) as demo:
    gr.Markdown(DESCRIPTION_MD)

    with gr.Tabs():
        with gr.TabItem("⚡ Manual Control"):
            with gr.Row():
                # Left panel — controls
                with gr.Column(scale=1):
                    gr.Markdown("### ⚡ Quick Start")
                    diff_dd = gr.Dropdown(choices=["Easy", "Medium", "Hard"], value="Easy", label="Difficulty")
                    act_dd = gr.Dropdown(choices=["ignore", "flag", "fix", "escalate"], value="flag", label="Action")
                    with gr.Row():
                        btn_reset = gr.Button("🔄 Reset", variant="secondary")
                        btn_step = gr.Button("▶ Step", variant="primary")
                        btn_state = gr.Button("📊 Get State")
                    
                    reward_disp = gr.Markdown("**Reward:** N/A | **Done:** False", elem_classes=["reward-badge"])
                    status_disp = gr.Textbox(label="Status", value="Ready.", interactive=False, lines=2)

                # Right panel — output
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 Raw JSON Response")
                    json_out = gr.Code(label="", language="json", value='{}', lines=18, elem_classes=["json-output"])

            btn_reset.click(handle_reset, [diff_dd], [status_disp, reward_disp, json_out])
            btn_step.click(handle_step, [act_dd], [status_disp, reward_disp, json_out])
            btn_state.click(handle_get_state, [], [status_disp, reward_disp, json_out])

        with gr.TabItem("🔍 Run AI Security Audit"):
            gr.Markdown("### 🔍 Live Agent Inference\nRuns the LLM agent through the environment tasks.")
            with gr.Row():
                with gr.Column(scale=1):
                    a_token = gr.Textbox(label="🔑 HF_TOKEN", placeholder="hf_...", type="password")
                    a_diff = gr.Dropdown(choices=["easy", "medium", "hard", "all"], value="easy", label="Difficulty")
                    btn_audit = gr.Button("🚀 Run Audit", variant="primary", size="lg")
                with gr.Column(scale=2):
                    a_logs = gr.Textbox(label="📡 Live Logs", lines=20, interactive=False, elem_classes=["audit-terminal"])
            btn_audit.click(run_security_audit, [a_diff, a_token], [a_logs])

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(): gr.Markdown(CONTRIBUTE_MD)
        with gr.Column(): gr.Markdown(README_MD)

# ── MOUNT AT ROOT ──
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
