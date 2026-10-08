# NVIDIA CUDA 12.6 development environment
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Install Python 3.12 and basic development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Tell Triton explicitly which CUDA toolkit and assembler to use
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV TRITON_PTXAS_PATH=/usr/local/cuda-12.6/bin/ptxas
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Set the working directory inside the container
WORKDIR /workspace/PRX
# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /usr/local/bin/

# Copy dependency definitions first
COPY pyproject.toml uv.lock ./

# Install dependencies exactly as recorded in uv.lock
RUN uv sync --frozen --no-install-project \
    --extra streaming \
    --extra lpips

# Copy PRX source code into the image
COPY . .

# Install PRX itself without changing locked dependencies
RUN uv sync --frozen \
    --extra streaming \
    --extra lpips

# Use the project's virtual environment by default
ENV PATH="/workspace/PRX/.venv/bin:${PATH}"

# Verify compiler and Python environment during the build
RUN nvcc --version && \
    ptxas --version && \
    python -c "import torch, triton; print('PyTorch:', torch.__version__); print('Torch CUDA:', torch.version.cuda); print('Triton:', triton.__version__)"
