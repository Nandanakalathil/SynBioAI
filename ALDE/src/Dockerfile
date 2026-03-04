FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    python3-dev \
    gcc \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /opt/program

# Install Python dependencies
COPY alde.yml .
RUN pip install --no-cache-dir \
    pandas==1.5 \
    numpy \
    matplotlib \
    seaborn \
    scipy \
    tqdm \
    openpyxl \
    pyro-ppl==1.8.6 \
    gpytorch==1.11 \
    botorch==0.9.4 \
    xgboost \
    boto3 \
    flask \
    gevent \
    gunicorn

# Copy the source code
COPY . .

# Set the entry point (SageMaker uses /opt/program/serve or /opt/ml/model/code/inference.py)
# For a custom container, we provide a serve script or use gunicorn directly.
ENTRYPOINT ["python", "serve.py"]
