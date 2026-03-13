# CatPred Async Inference

This folder contains the standalone tool for running enzyme kinetics predictions using the CatPred SageMaker asynchronous endpoint.

## Prerequisites

- **Python 3.10+** (tested on python3.10)
- **AWS Credentials**: Configured via `aws configure` or environment variables.
- **Dependencies**: 
  ```bash
  pip install boto3 pandas
  ```

## Usage

You can run predictions using the `predict.py` script. The script automatically handles data conversion, S3 uploads, endpoint invocation, and result retrieval.

### Basic command
If you have your sequence and SMILES in a CSV, just provide the path:
```bash
python3.10 predict.py path/to/your_data.csv
```
*Note: By default, this predicts **kcat** and saves the result to `your_data_result.csv`.*

### Predicting different parameters
The model supports Turnover Frequency ($k_{cat}$), Michaelis Constant ($K_m$), and Inhibition Constant ($K_i$).
```bash
# Predict Michaelis Constant (Km)
python3.10 predict.py data.csv --km

# Predict Inhibition Constant (Ki)
python3.10 predict.py data.csv --ki
```
*The script automatically selects the correct model weights on the server when you change the parameter.*

### Customizing the output
Use the `-o` or `--output` flag to specify a custom filename for the results:
```bash
python3.10 predict.py data.csv -o finalize_predictions.csv
```

---

## Input CSV Format
Your CSV should contain at least two columns:
1. `SMILES`: The chemical structure of the substrate.
2. `sequence`: The amino acid sequence of the enzyme.

*Structural data (PDB files) is **not required**. The script automatically generates necessary placeholders, and the model uses ESM2 embeddings for the prediction.*

## Troubleshooting
If you need to use a different S3 bucket or endpoint name than the defaults:
```bash
export S3_BUCKET="your-bucket-name"
export SM_ENDPOINT_NAME="your-endpoint-name"
python3.10 predict.py data.csv
```
