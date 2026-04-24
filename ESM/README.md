# ESM2 Embedding Service (SageMaker Asynchronous Endpoint)

This folder contains the implementation of a cloud-native ESM2 embedding extraction service using AWS SageMaker Asynchronous Inference. It is designed to handle large-scale protein sequence datasets by leveraging GPU instances (`ml.g5.xlarge`) and automatically scaling to zero when idle to save costs.

## Architecture

- **Model**: ESM2-650M (Evolutionary Scale Model)
- **Deployment**: SageMaker Asynchronous Endpoint
- **Scaling**: Auto-scaling configured to **Scale-to-Zero** after 15 minutes of inactivity.
- **Data Flow**: FASTA input -> S3 -> SageMaker Endpoint -> S3 Output -> Client.

## Repository Structure

| File | Description |
| :--- | :--- |
| `Dockerfile.esm` | Docker recipe for the GPU-optimized inference container. |
| `serve_esm.py` | Inference script (entry point) using Flask to serve embeddings. |
| `deploy_esm_async.py` | Main deployment script to build, push, and deploy to AWS. |
| `invoke_esm.py` | Client script to submit jobs and poll for results. |
| `utils.py` | Shared utilities for model loading and data processing. |
| `test.fasta` | Example FASTA file for validation. |

## Quick Start

### 1. Prerequisites

Ensure you have the following installed:
- Docker Desktop
- AWS CLI configured with appropriate credentials
- Python environment with `sagemaker`, `boto3`, `biopython`, and `pandas`.

```bash
pip install sagemaker boto3 biopython pandas
```

### 2. Deploy to AWS

Run the deployment script. This will build the Docker image, push it to Amazon ECR, and create the SageMaker model, endpoint configuration, and asynchronous endpoint.

```bash
python deploy_esm_async.py
```

*Note: The script handles cleaning up any existing resources with the same name before deploying.*

### 3. Generate Embeddings

Use `invoke_esm.py` to submit a FASTA file for processing.

**Asynchronous Invocation (Recommended for large files):**
```bash
python invoke_esm.py --fasta test.fasta --endpoint_name ESM2-650M-Endpoint-Async --is_async --batch_size 16
```

**Synchronous Invocation (For small batches):**
```bash
python invoke_esm.py --fasta test.fasta --endpoint_name ESM2-650M-Endpoint-Async --batch_size 4
```

*Note: The `--batch_size` parameter determines how many sequences are sent in a single request. For the 650M model, a batch size of 16-32 is typically optimal for GPU memory.*

## Auto-Scaling Behavior

The endpoint is configured with a **Scale-to-Zero** policy:
- **Max Capacity**: 2 instances.
- **Min Capacity**: 0 instances.
- **Scale-In Cooldown**: 900 seconds (15 minutes).

If no requests are sent for 15 minutes, the instance will be terminated. When a new request arrives, SageMaker will automatically spin up a new instance, process the request, and store the result in S3.

## Local Testing

You can test the container locally before deploying:

1. **Build the image**:
   ```bash
   docker build -t esm2-local -f Dockerfile.esm .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8080:8080 esm2-local
   ```

3. **Invocations**:
   ```bash
   python invoke_esm.py --fasta test.fasta --local_url http://localhost:8080/invocations
   ```

## Output Format
The service returns a CSV file where:
- The index column contains the sequence IDs from the FASTA file.
- The remaining columns represent the mean-pooled ESM2 embeddings (1280 dimensions for the 650M model).
