FROM python:3.10-slim

WORKDIR /app

# Install uv for fast installation
RUN pip install uv

# Copy dependency files first (better Docker layer caching)
COPY pyproject.toml .
COPY server/requirements.txt server/requirements.txt

# Install dependencies using uv
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install .

# Copy remaining project files
COPY . .

# Set environment variables for uv and execution
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Expose the default server port
EXPOSE 7860

# Run openenv GUI server by default
CMD ["python", "gradio_app.py"]