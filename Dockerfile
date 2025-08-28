FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       curl \
       ca-certificates \
       git \
       build-essential \
       libgl1 \
       libglib2.0-0 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (https://docs.astral.sh/uv/)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Use uv-managed Python
ARG UV_PYTHON_VERSION=3.11
ENV UV_PYTHON=${UV_PYTHON_VERSION}
RUN uv python install ${UV_PYTHON_VERSION}

WORKDIR /app

# Clone repository instead of copying local workspace (avoids large data in image)
RUN git clone --depth 1 --filter=blob:none --single-branch https://github.com/creebin4/Salamander-Software /app

# Install project dependencies using uv; prefer lockfile if present
RUN if [ -f uv.lock ]; then \
      uv sync --frozen --python ${UV_PYTHON_VERSION} --no-dev; \
    else \
      uv sync --python ${UV_PYTHON_VERSION} --no-dev; \
    fi

# Install IBEIS (Linux-only)
RUN uv add ibeis

ENV PYTHONUNBUFFERED=1

# Run the training/inference script via uv
CMD ["uv", "run", "--python", "3.11", "python", "main.py"]


