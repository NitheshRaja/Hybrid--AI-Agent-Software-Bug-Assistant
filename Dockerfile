# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install only minimal runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

ADD . /app

# uv is configured in pyproject.toml to fetch:
# - llama-cpp-python: pre-built CPU binary (no C++ compilation)
# - torch: CPU-only wheel (avoids ~2GB of CUDA packages)
RUN uv sync --no-install-project --no-dev

ENV PORT=7860
EXPOSE 7860

# Launch the app immediately; the Gemma download will run in the background
CMD ["sh", "-c", "uv run python web_ui.py"]
