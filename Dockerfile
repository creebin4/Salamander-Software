FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS base

# System dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       curl \
       ca-certificates \
       git \
       libgl1 \
       libglib2.0-0 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Python via uv
ARG UV_PYTHON_VERSION=3.11
RUN uv python install ${UV_PYTHON_VERSION}

# Use a central venv directory (not inside /app)
ENV UV_VENV_DIR=/root/.venvs

WORKDIR /app

# Shallow clone repo (no history, no large files)
RUN git clone --depth 1 --filter=blob:none --single-branch https://github.com/creebin4/Salamander-Software /app

# Install project dependencies
RUN if [ -f uv.lock ]; then \
      uv sync --frozen --python ${UV_PYTHON_VERSION} --no-dev --no-cache-dir; \
    else \
      uv sync --python ${UV_PYTHON_VERSION} --no-dev --no-cache-dir; \
    fi \
    && uv add ibeis \
    && rm -rf /root/.cache

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "--python", "3.11", "python", "main.py"]