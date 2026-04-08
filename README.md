---

title: AppSec Agent
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false

---

# AI Security Code Reviewer — AppSec Agent RL Environment

> A production-grade **Reinforcement Learning environment** built on the
> [OpenEnv](https://github.com/openenv) framework that simulates a real-world
> Application Security (AppSec) code-review pipeline.

An AI agent analyses code snippets flagged by static-analysis tooling and
decides the optimal response for each finding:

| Action     | Meaning                                              |
|------------|------------------------------------------------------|
| `ignore`   | False positive — no action needed                    |
| `flag`     | Real issue — needs developer attention               |
| `fix`      | Real issue — can be auto-remediated safely           |
| `escalate` | Critical issue — requires human security expert      |

Rewards penalise over-flagging, missed critical vulnerabilities, repetitive
(loop) behaviour, and reward efficient, accurate triage.

---

## 📁 Project Structure

```
openev-project/
├── Dockerfile              # Slim Docker image for deployment
├── README.md               # This file
├── pyproject.toml          # Project metadata & dependencies (PEP 621)
├── uv.lock                 # Pinned dependency lock file (uv)
├── openenv.yaml            # OpenEnv environment specification
├── models.py               # Pydantic models: Observation, Action, Reward
├── inference.py            # LLM-powered inference loop (HuggingFace router)
├── client.py               # Typed Python client (OpenEnv EnvClient)
├── __init__.py             # Root package marker
├── server/
│   ├── __init__.py         # Server package marker
│   ├── app.py              # FastAPI application (OpenEnv HTTP server)
│   ├── environment.py      # Core RL environment logic & reward table
│   └── requirements.txt    # Pinned pip dependencies for the server
└── tasks/
    ├── easy.json           # 3 unambiguous scenarios
    ├── medium.json         # 5 mixed scenarios with false positives
    └── hard.json           # 7 highly ambiguous scenarios
```

---

## ⚡ Quick Start

### Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) **or** pip + venv
- A HuggingFace API token (`HF_TOKEN`)

### 1. Clone & Install

```bash
git clone <repo-url> openev-project && cd openev-project

# Option A — uv (recommended)
uv venv && uv pip install -e ".[dev]"

# Option B — pip
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Set Environment Variables

```bash
export HF_TOKEN="hf_your_token_here"

# Optional overrides
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export TASK_DIFFICULTY="easy"     # easy | medium | hard
```

### 3. Run the Environment Server

```bash
# Using uv
uv run server

# Using uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or directly
python -m server.app --port 8000
```

### 4. Run Inference

```bash
python inference.py
```

The inference script runs all three difficulty tiers (`easy → medium → hard`)
and prints structured logs in the strict OpenEnv format:

```
[START]
# task=appsec-code-review-easy env=appsec-openenv-v1 model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=fix reward=0.8500 done=false error=null
[STEP] step=2 action=escalate reward=1.0000 done=false error=null
[STEP] step=3 action=ignore reward=1.0000 done=true error=null
[END] success=true steps=3 score=0.9500 rewards=0.8500,1.0000,1.0000
```

---

## 🐳 Docker

```bash
# Build
docker build -t appsec-env:latest .

# Run
docker run --rm -e HF_TOKEN="hf_..." appsec-env:latest

# Run the server instead of inference
docker run --rm -p 8000:8000 -e HF_TOKEN="hf_..." appsec-env:latest \
    uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## 🧩 Environment Details

### Observation Schema

| Field            | Type   | Description                                    |
|------------------|--------|------------------------------------------------|
| `code_snippet`   | string | The code block under review                    |
| `detected_issue` | string | Static analysis finding / vulnerability desc.  |
| `severity`       | string | `low` \| `medium` \| `high` \| `critical`     |
| `context`        | string | Deployment context, exposure, risk amplifiers  |
| `step_count`     | int    | Current step within the episode                |

### Action Schema

| Field    | Type   | Description                                      |
|----------|--------|--------------------------------------------------|
| `action` | string | `ignore` \| `flag` \| `fix` \| `escalate`       |

### Reward Schema

| Field       | Type   | Description                                     |
|-------------|--------|-------------------------------------------------|
| `value`     | float  | Normalised reward in `[0.0, 1.0]`               |
| `reasoning` | string | Deterministic explanation of the reward          |

### Task Tiers

| Tier   | Scenarios | Max Steps | Description                           |
|--------|-----------|-----------|---------------------------------------|
| easy   | 3         | 3         | Unambiguous, textbook vulnerabilities |
| medium | 5         | 5         | Mixed real + false positives          |
| hard   | 7         | 7         | Highly ambiguous, conflicting signals |

---

## 🧪 Testing

```bash
# Run all tests
pytest -v

# With coverage
pytest --cov=server --cov=models -v
```

---

## 📝 Environment Variables

| Variable         | Required | Default                                  | Description                      |
|------------------|----------|------------------------------------------|----------------------------------|
| `HF_TOKEN`       | ✅ Yes   | —                                        | HuggingFace API token            |
| `API_BASE_URL`   | No       | `https://router.huggingface.co/v1`       | OpenAI-compatible endpoint       |
| `MODEL_NAME`     | No       | `Qwen/Qwen2.5-7B-Instruct`              | LLM model identifier             |
| `TASK_DIFFICULTY` | No      | `easy`                                   | Task tier: easy / medium / hard  |

---

## 📄 License

This project was developed for the Meta OpenEnv Hackathon 2026.
