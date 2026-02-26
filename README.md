# SynBioAI SageMaker Endpoints

This repository contains the inference scripts and documentation for deploying and using protein engineering models as SageMaker asynchronous endpoints.

## Models Available

| Model | Folder | Endpoint Name | 
|-------|--------|---------------|
| **S3F** | `s3f/` | `s3f-endpoint-async-v3` |
| **ESM Scan** | `esm-scan/` | `esm-scan-endpoint-async` | 
| **ProSST** | `prosst/` | `prosst-inference-endpoint` | 
| **DeepProzyme** | `deepec/` | `deepprozyme-endpoint` | 
| **VespaG** | `vespag/` | `vespag-endpoint` | 
| **ALDE** | `alde/` | `vespag-endpoint` | 

## Setup

1. **Install Prerequisites**:
   ```bash
   pip install awscli boto3
   ```

2. **Configure AWS**:
   Enter credentials when prompted (refer to credentials.txt)

   ```bash
   aws configure
   ```
## General Management

For detailed instructions on each model, refer to the `README.md` inside its specific folder.

## Maintenance & Cleanup

### 1. Cleanup S3 Storage
The inference scripts upload files to S3. To clear temporary files and save storage costs:
```bash
chmod +x cleanup-s3.sh
./cleanup-s3.sh
```

Note: All endpoints automatically scale to zero after 15 minutes of inactivity.
