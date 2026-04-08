"""
app.py — AppSec Agent: 1:1 Screenshot Match Integration
========================================================
Matches the user-provided screenshot exactly while maintaining 
the fixed FastAPI/OpenEnv backend for HF Spaces.
"""

import os
import sys
import json
import queue
import threading
from pathlib import Path
from typing import Generator, Optional

# ── sys.path bootstrap ─────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── FastAPI / OpenEnv Backend ──────────────────────────────────────────────────
try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError("openenv-core is required. Install with: pip install openenv-core") from exc

from fastapi import FastAPI
from models import MyAction, MyObservation
from server.environment import MyEnvironment, AppSecEnvironment
from models import AppSecAction

app: FastAPI = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="appsec_code_reviewer",
    max_concurrent_envs=4,
)

# ── Gradio Logic ───────────────────────────────────────────────────────────────
import gradio as gr

_env: Optional[AppSecEnvironment] = None
_last_obs = None

def handle_reset(difficulty: str):
    global _env, _last_obs
    _env = AppSecEnvironment(task_difficulty=difficulty.lower())
    _last_obs = _env.reset()
    status = "✅ Environment reset successfully!"
    reward_info = "Reward: 0.0000 | Done: False"
    return status, reward_info, json.dumps({"observation": _last_obs.model_dump()}, indent=2)

def handle_step(action: str):
    global _env, _last_obs
    if _env is None: return "⚠️ Reset first!", "", "{}"
    try:
        action_obj = AppSecAction(action=action.lower())
        _last_obs, reward, done, _ = _env.step(action_obj)
        status = f"✅ Step complete! {'🏁 Done!' if done else ''}"
        reward_info = f"Reward: {reward.value:.4f} | Done: {done}"
        return status, reward_info, json.dumps({"observation": _last_obs.model_dump(), "reward": reward.value, "done": done}, indent=2)
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "{}"

def handle_get_state():
    global _env, _last_obs
    if _env is None or _last_obs is None: return "⚠️ No session.", "", "{}"
    return "✅ State retrieved!", f"Step: {_last_obs.step_count}", json.dumps({"observation": _last_obs.model_dump()}, indent=2)

def run_security_audit(difficulty: str, hf_token: str) -> Generator[str, None, None]:
    from openai import OpenAI
    log_queue = queue.Queue(); accumulated = []
    def _run():
        token = hf_token.strip() or os.environ.get("HF_TOKEN", "")
        if not token: log_queue.put("❌ ERROR: HF_TOKEN required.\n"); log_queue.put("__DONE__"); return
        client = OpenAI(base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"), api_key=token)
        tasks = [difficulty.lower()] if difficulty.lower() != "all" else ["easy", "medium", "hard"]
        for d in tasks:
            e = AppSecEnvironment(task_difficulty=d); log_queue.put(f"[START]\n# task={d} env=appsec-v1\n")
            o = e.reset(); done = False; s = 0
            while not done:
                s += 1
                try:
                    res = client.chat.completions.create(model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"), messages=[{"role":"user","content":str(o)}], temperature=0.1, max_tokens=100)
                    a = "flag"
                    if "ACTION:" in (res.choices[0].message.content or "").upper(): a = (res.choices[0].message.content or "").split("ACTION:")[-1].split()[0].strip().lower()
                    o, r, done, _ = e.step(AppSecAction(action=a))
                    log_queue.put(f"[STEP] step={s} action={a} reward={r.value:.4f} done={str(done).lower()}\n")
                except: break
            log_queue.put(f"[END] score={e.compute_final_score():.4f}\n\n")
        log_queue.put("__DONE__")
    threading.Thread(target=_run, daemon=True).start()
    yield "🚀 Starting Audit...\n"
    while True:
        l = log_queue.get()
        if l == "__DONE__": break
        accumulated.append(l); yield "".join(accumulated)

# ── UI Blocks ─────────────────────────────────────────────────────────────────

CSS = """
.container { max-width: 1200px; margin: auto; }
.json-output { font-family: 'Courier New', monospace; font-size: 13px; background: #0b0e14 !important; }
.reward-badge { font-weight: bold; color: #22c55e; font-size: 14px; margin-top: 10px; }
.status-box { background: #1a1e26; border: 1px solid #30363d; border-radius: 6px; padding: 10px; }
.audit-log { font-family: 'Courier New', monospace; font-size: 11px; background: #0d1117; color: #58a6ff; }
footer { display: none !important; }
"""

with gr.Blocks(title="AppSec Agent — AI Security Code Reviewer", theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"), css=CSS) as demo:
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
        # Left: Quick Start
        with gr.Column(scale=1):
            gr.Markdown("### ⚡ Quick Start")
            diff_dd = gr.Dropdown(choices=["Easy", "Medium", "Hard"], value="Easy", label="Task Difficulty")
            act_dd = gr.Dropdown(choices=["ignore", "flag", "fix", "escalate"], value="flag", label="Action")
            
            with gr.Row():
                btn_step = gr.Button("▶ Step", variant="primary")
                btn_reset = gr.Button("🔄 Reset", variant="secondary")
                btn_state = gr.Button("📊 Get State")
            
            reward_disp = gr.Markdown("**Reward:** N/A | **Done:** False", elem_classes=["reward-badge"])
            status_disp = gr.Textbox(label="Status", value="Ready.", interactive=False, lines=2, elem_classes=["status-box"])

        # Right: JSON
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Raw JSON Response")
            json_out = gr.Code(label="", language="json", interactive=False, lines=15, elem_classes=["json-output"])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🤝 Contribute to this environment")
            gr.Markdown("Submit improvements via pull request:\n`openenv fork namantiwari/appsec-agent --hf`")
        with gr.Column():
            gr.Markdown("### 📖 README")
            gr.Markdown("**Task Difficulties:** 🟢 Easy • 🟡 Medium • 🔴 Hard\nRewards range 0.0 – 1.0. Final score ≥ 0.60 is success.")

    with gr.Accordion("🔍 AI Security Audit (LLM Agent)", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                a_token = gr.Textbox(label="HF_TOKEN", type="password")
                a_diff = gr.Dropdown(choices=["easy", "medium", "hard", "all"], value="easy", label="Difficulty")
                btn_audit = gr.Button("🚀 Run Audit", variant="primary")
            with gr.Column(scale=2):
                a_logs = gr.Textbox(label="Audit Logs", lines=10, interactive=False, elem_classes=["audit-log"])

    btn_reset.click(handle_reset, [diff_dd], [status_disp, reward_disp, json_out])
    btn_step.click(handle_step, [act_dd], [status_disp, reward_disp, json_out])
    btn_state.click(handle_get_state, [], [status_disp, reward_disp, json_out])
    btn_audit.click(run_security_audit, [a_diff, a_token], [a_logs])

app = gr.mount_gradio_app(app, demo, path="/") # MOUNT AT ROOT

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
