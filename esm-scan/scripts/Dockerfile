# ESM-Scan Docker Image for SageMaker Async Inference
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/conda/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir \
    torch==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other Python packages
RUN pip install --no-cache-dir \
    fair-esm \
    biopython \
    numpy \
    pandas \
    flask \
    gunicorn

# Create directories
RUN mkdir -p /opt/ml/model /opt/ml/code

# Pre-download the ESM-1v model weights during build (wget only, no Python loading)
RUN wget -q --show-progress -O /opt/ml/model/esm1v_t33_650M_UR90S_1.pt \
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt" && \
    ls -lh /opt/ml/model/esm1v_t33_650M_UR90S_1.pt

# Copy inference code
COPY source/inference.py /opt/ml/code/inference.py
COPY source/serve.py /opt/ml/code/serve.py

# Copy reference data files
COPY source/benign_scores.npy /opt/ml/code/benign_scores.npy
COPY source/pathogenic_scores.npy /opt/ml/code/pathogenic_scores.npy

EXPOSE 8080

# SageMaker requires the container to run a web server on port 8080
# with /ping and /invocations endpoints
ENTRYPOINT ["python", "/opt/ml/code/serve.py"]
