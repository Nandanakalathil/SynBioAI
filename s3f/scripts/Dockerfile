# S3F Docker Image (Manual CUDA Install to bypass limits)
FROM public.ecr.aws/ubuntu/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates git build-essential gnupg \
    libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA 11.8 manually (bypassing Docker Hub images)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-11-8 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda (Python 3.10)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir \
    torch==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric extensions
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.5.1+cu118.html && \
    pip install --no-cache-dir torch-geometric

# Install torchdrug and other dependencies
RUN pip install --no-cache-dir \
    torchdrug \
    fair-esm \
    biopython \
    "numpy<2" \
    pandas \
    scipy \
    scikit-learn \
    easydict \
    pyyaml \
    jinja2 \
    flask \
    gunicorn \
    tqdm \
    potpourri3d \
    robust_laplacian \
    pykeops

# Create directories
RUN mkdir -p /opt/ml/model /opt/ml/code /opt/s3f

# Clone S3F repository
RUN git clone https://github.com/DeepGraphLearning/S3F.git /opt/s3f/S3F

# Download S3F model checkpoint
RUN wget -q --show-progress -O /opt/ml/model/s3f.pth \
    "https://zenodo.org/records/14257708/files/s3f.pth?download=1"

# Download ESM-2-650M model weights
RUN mkdir -p /opt/ml/model/esm-model-weights && \
    wget -q --show-progress -O /opt/ml/model/esm-model-weights/esm2_t33_650M_UR50D.pt \
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"

# Copy inference code
COPY source/serve.py /opt/ml/code/serve.py

# Add S3F to Python path
ENV PYTHONPATH="/opt/s3f/S3F:${PYTHONPATH}"

EXPOSE 8080
ENTRYPOINT ["python", "/opt/ml/code/serve.py"]
