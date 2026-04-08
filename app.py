"""
app.py — AppSec Agent: Unified FastAPI + Gradio Root Entry Point
================================================================
Runs the OpenEnv-compliant FastAPI backend AND the Gradio dashboard
on the same port (7860) with Gradio mounted at the ROOT path.

Exposes:
  /          → Gradio UI
  /reset     → OpenEnv reset (FastAPI)
  /step      → OpenEnv step (FastAPI)
  /state     → OpenEnv state (FastAPI)
"""

import os
import sys
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
    raise ImportError("openenv-core is required. Install with: pip install openenv-core") from exc

from fastapi import FastAPI
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

# ── Gradio Dashboard ───────────────────────────────────────────────────────────
import json
import gradio as gr
from server.environment import AppSecEnvironment
from models import AppSecAction

# ── Global env state ──────────────────────────────────────────────────────────
_env: Optional[AppSecEnvironment] = None
_last_obs = None

def handle_reset(difficulty: str):
    global _env, _last_obs
    _env = AppSecEnvironment(task_difficulty=difficulty.lower())
    _last_obs = _env.reset()
    obs_dict = _last_obs.model_dump()
    status = "✅ Environment reset successfully!"
    reward_info = "Reward: N/A | Done: False"
    return (status, reward_info, json.dumps({"observation": obs_dict}, indent=2))

def handle_step(action: str):
    global _env, _last_obs
    if _env is None:
        return ("⚠️ Reset first!", "", json.dumps({"error": "Init first."}, indent=2))
    try:
        action_obj = AppSecAction(action=action.lower())
        _last_obs, reward, done, info = _env.step(action_obj)
        obs_dict = _last_obs.model_dump()
        status = f"✅ Step complete! {'🏁 Done!' if done else ''}"
        reward_info = f"Reward: {reward.value:.4f} | Done: {done}"
        return status, reward_info, json.dumps({"obs": obs_dict, "reward": reward.value, "done": done}, indent=2)
    except Exception as e:
        return "❌ Error", "", json.dumps({"error": str(e)}, indent=2)

def handle_get_state():
    global _env, _last_obs
    if _env is None or _last_obs is None:
        return ("⚠️ Reset first!", "", json.dumps({"error": "No session."}, indent=2))
    return ("✅ State retrieved!", f"Step: {_last_obs.step_count}", json.dumps({"obs": _last_obs.model_dump()}, indent=2))

# ── Live Audit Logic ──────────────────────────────────────────────────────────
def run_security_audit(difficulty: str, hf_token: str) -> Generator[str, None, None]:
    import queue, threading
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
                        messages=[{"role":"system","content":"Respond with ACTION: <ignore|flag|fix|escalate>"},{"role":"user","content":str(obs)}],
                        temperature=0.1, max_tokens=100
                    )
                    act_str = "flag"
                    for line in (res.choices[0].message.content or "").splitlines():
                        if "ACTION:" in line.upper(): act_str = line.split(":")[-1].strip().lower()
                    obs, reward, done, _ = env.step(AppSecAction(action=act_str))
                    log_queue.put(f"[STEP] step={step} action={act_str} reward={reward.value:.4f} done={str(done).lower()}\n")
                except Exception as e:
                    log_queue.put(f"ERROR: {str(e)}\n")
                    break
            log_queue.put(f"[END] score={env.compute_final_score():.4f}\n\n")
        log_queue.put("__DONE__")

    threading.Thread(target=_run, daemon=True).start()
    yield "🚀 Starting Audit...\n"
    while True:
        line = log_queue.get()
        if line == "__DONE__": break
        accumulated.append(line)
        yield "".join(accumulated)

# ── UI Layout ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="AppSec Agent", theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate")) as demo:
    gr.Markdown("# 🛡️ AppSec Agent — AI Code Reviewer")
    with gr.Tabs():
        with gr.TabItem("⚡ Manual Control"):
            with gr.Row():
                with gr.Column():
                    difficulty = gr.Dropdown(choices=["Easy", "Medium", "Hard"], value="Easy", label="Difficulty")
                    action = gr.Dropdown(choices=["ignore", "flag", "fix", "escalate"], value="flag", label="Action")
                    with gr.Row():
                        btn_reset = gr.Button("🔄 Reset", variant="secondary")
                        btn_step = gr.Button("▶ Step", variant="primary")
                    status = gr.Textbox(label="Status", interactive=False)
                with gr.Column():
                    json_out = gr.Code(label="JSON Response", language="json", lines=15)
            btn_reset.click(handle_reset, [difficulty], [status, gr.State(), json_out])
            btn_step.click(handle_step, [action], [status, gr.State(), json_out])
            
        with gr.TabItem("🔍 AI Security Audit"):
            token_input = gr.Textbox(label="HF_TOKEN", placeholder="hf_...", type="password")
            audit_diff = gr.Dropdown(choices=["easy", "medium", "hard", "all"], value="easy", label="Difficulty")
            btn_audit = gr.Button("🚀 Run Security Audit", variant="primary")
            audit_log = gr.Textbox(label="Audit Logs", lines=20, interactive=False)
            btn_audit.click(run_security_audit, [audit_diff, token_input], [audit_log])

# ── MOUNT AT ROOT / ──
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
