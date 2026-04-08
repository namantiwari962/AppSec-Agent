"""
server/app.py — AppSec Agent: Unified Server (Structural Restoration)
====================================================================
Unified FastAPI + Gradio entry point moved back to the server/ directory.
Aesthetics: 1:1 Match with User Screenshot.
Technical: Fixed 500 errors and pathing.
"""

import os
import sys
import json
import queue
import threading
from pathlib import Path
from typing import Generator, Optional, List

# ── sys.path bootstrap ─────────────────────────────────────────────────────────
# Crucial for finding root-level modules like models.py
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Global Environment Variables (Fixes 500 Errors) ──────────────────────────
# Defined globally as requested by the user
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None # Fallback handled in audit function

# ── FastAPI / OpenEnv Backend ──────────────────────────────────────────────────
try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError("openenv-core is required. Install with: pip install openenv-core") from exc

from fastapi import FastAPI
from models import MyAction, MyObservation
# Use absolute import relative to root (since root is in sys.path)
from server.environment import MyEnvironment, AppSecEnvironment
from models import AppSecAction

# Initialize the OpenEnv backend app
app: FastAPI = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="appsec_code_reviewer",
    max_concurrent_envs=4,
)

# ── Gradio Logic ───────────────────────────────────────────────────────────────
import gradio as gr

_env_session: Optional[AppSecEnvironment] = None
_last_obs_session = None

def handle_reset(difficulty: str):
    global _env_session, _last_obs_session
    _env_session = AppSecEnvironment(task_difficulty=difficulty.lower())
    _last_obs_session = _env_session.reset()
    status = "✅ Environment reset successfully!"
    reward_info = "Reward: 0.0000 | Done: False"
    return status, reward_info, json.dumps({"observation": _last_obs_session.model_dump()}, indent=2)

def handle_step(action: str):
    global _env_session, _last_obs_session
    if _env_session is None: return "⚠️ Please Reset first!", "", "{}"
    try:
        action_obj = AppSecAction(action=action.lower())
        _last_obs_session, reward, done, info = _env_session.step(action_obj)
        status = f"✅ Step complete! {'🏁 Done!' if done else ''}"
        reward_info = f"Reward: {reward.value:.4f} | Done: {done}"
        return status, reward_info, json.dumps({"obs": _last_obs_session.model_dump(), "reward": reward.value, "done": done}, indent=2)
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "{}"

def handle_get_state():
    global _env_session, _last_obs_session
    if _env_session is None or _last_obs_session is None: return "⚠️ No session.", "", "{}"
    return "✅ State retrieved!", f"Step: {_last_obs_session.step_count}", json.dumps({"obs": _last_obs_session.model_dump()}, indent=2)

def run_security_audit(difficulty: str, user_token: str) -> Generator[str, None, None]:
    log_queue = queue.Queue(); accumulated = []
    def _run():
        token = user_token.strip() or HF_TOKEN
        if not token:
            log_queue.put("❌ ERROR: HF_TOKEN is required.\n")
            log_queue.put("__DONE__"); return
        if OpenAI is None:
            log_queue.put("❌ ERROR: OpenAI library not found.\n")
            log_queue.put("__DONE__"); return
            
        client = OpenAI(base_url=API_BASE_URL, api_key=token)
        tasks = [difficulty.lower()] if difficulty.lower() != "all" else ["easy", "medium", "hard"]
        
        for d in tasks:
            env = AppSecEnvironment(task_difficulty=d)
            log_queue.put(f"[START]\n# task=appsec-{d} env=appsec-v1 model={MODEL_NAME}\n")
            obs = env.reset()
            done = False; step = 0
            while not done:
                step += 1
                try:
                    res = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role":"system","content":"Respond with ACTION: <ignore|flag|fix|escalate>"},{"role":"user","content":str(obs)}],
                        temperature=0.1, max_tokens=150
                    )
                    act_str = "flag"
                    content = res.choices[0].message.content or ""
                    for line in content.splitlines():
                        if "ACTION:" in line.upper(): act_str = line.split(":")[-1].strip().lower()
                    obs, reward, done, _ = env.step(AppSecAction(action=act_str))
                    log_queue.put(f"[STEP] step={step} action={act_str} reward={reward.value:.4f} done={str(done).lower()}\n")
                except Exception as e:
                    log_queue.put(f"ERROR: {str(e)}\n"); break
            log_queue.put(f"[END] score={env.compute_final_score():.4f}\n\n")
        log_queue.put("__DONE__")

    threading.Thread(target=_run, daemon=True).start()
    yield "🚀 Starting Security Audit...\n\n"
    while True:
        line = log_queue.get()
        if line == "__DONE__": break
        accumulated.append(line); yield "".join(accumulated)

# ── UI Aesthetics (Original Screenshot Match) ──────────────────────────────────

with gr.Blocks(
    title="AppSec Agent — AI Security Code Reviewer",
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"),
    css="""
    .container { max-width: 1100px; margin: auto; }
    .json-output { font-family: 'Courier New', monospace; font-size: 13px; background: #0b1117 !important; }
    .reward-badge { font-weight: bold; color: #22c55e; font-size: 15px; }
    .status-box { background: #1a1e26; border: 1px solid #30363d; padding: 10px; border-radius: 6px; }
    footer { display: none !important; }
    """
) as demo:
    gr.Markdown("# 🛡️ AppSec Agent — AI Security Code Reviewer")
    gr.Markdown("An **OpenEnv-compatible** Reinforcement Learning environment for Application Security. The agent reviews code snippets and decides the optimal security action: `ignore` • `flag` • `fix` • `escalate`")

    with gr.Accordion("🔗 Connect to this environment", open=True):
        gr.Markdown("""
```python
from openenv.core import MyEnvAction, MyEnvObservation
with MyEnvClient(base_url="https://namantiwari-appsec-agent.hf.space") as env:
    result = await env.reset()
```
Or connect directly:  
`env = MyEnvClient(base_url="https://namantiwari-appsec-agent.hf.space")`
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚡ Quick Start")
            diff_dd = gr.Dropdown(choices=["Easy", "Medium", "Hard"], value="Easy", label="Task Difficulty")
            act_dd = gr.Dropdown(choices=["ignore", "flag", "fix", "escalate"], value="flag", label="Action")
            with gr.Row():
                btn_step = gr.Button("▶ Step", variant="primary")
                btn_reset = gr.Button("🔄 Reset", variant="secondary")
                btn_state = gr.Button("📊 Get State")
            reward_disp = gr.Markdown("**Reward:** N/A | **Done:** False", elem_classes=["reward-badge"])
            status_disp = gr.Textbox(label="Status", value="Ready.", interactive=False, lines=2)

        with gr.Column(scale=1):
            gr.Markdown("### 📋 Raw JSON Response")
            json_out = gr.Code(label="", language="json", interactive=False, lines=15, elem_classes=["json-output"])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🤝 Contribute to this environment")
            gr.Markdown("Submit via pull request on HF Hub:\n`openenv fork namantiwari/appsec-agent --hf`")
        with gr.Column():
            gr.Markdown("### 📖 README")
            gr.Markdown("**Difficulties:** 🟢 Easy • 🟡 Medium • 🔴 Hard\nRewards range 0.0 – 1.0. Final score ≥ 0.60 is success.")

    with gr.Accordion("🔍 AI Security Audit (LLM Agent)", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                a_token = gr.Textbox(label="HF_TOKEN", placeholder="hf_...", type="password")
                a_diff = gr.Dropdown(choices=["easy", "medium", "hard", "all"], value="easy", label="Difficulty")
                btn_audit = gr.Button("🚀 Run Audit", variant="primary")
            with gr.Column(scale=2):
                a_logs = gr.Textbox(label="Audit Logs", lines=10, interactive=False)

    btn_reset.click(handle_reset, [diff_dd], [status_disp, reward_disp, json_out])
    btn_step.click(handle_step, [act_dd], [status_disp, reward_disp, json_out])
    btn_state.click(handle_get_state, [], [status_disp, reward_disp, json_out])
    btn_audit.click(run_security_audit, [a_diff, a_token], [a_logs])

# ── MOUNT GRADIO LAST ──────────────────────────────────────────────────────────
# Mounting at root / ensures immediate accessibility on HF Spaces
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
