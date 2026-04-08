# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/lists/*

# ── Install uv (fast pip replacement) ─────────────────────────────────────────
RUN pip install --no-cache-dir uv

# ── Copy dependency files first (maximise Docker layer cache hits) ─────────────
COPY pyproject.toml .

# ── Create venv and install all project dependencies ──────────────────────────
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install ".[dev]"

# ── Copy the rest of the project ───────────────────────────────────────────────
COPY . .

# ── Configure environment ──────────────────────────────────────────────────────
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Change ownership to non-root user
RUN chown -R user:user /app /opt/venv
USER user

# ── Networking ─────────────────────────────────────────────────────────────────
# Hugging Face Spaces requires port 7860
EXPOSE 7860

# ── Launch: unified FastAPI + Gradio server on 0.0.0.0:7860 ───────────────────
CMD ["python", "server/app.py"]