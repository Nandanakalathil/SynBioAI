# ESM Scan: Variant Effect Prediction

ESM Scan uses the ESM-1v (single) transformer model with 650M parameters to predict the functional impact of protein mutations. It supports both full-sequence scans and specific mutation scoring.

## 1. Run Inference

```bash
python3 esm-scan/esm-scan-inference.py \
  -i esm-scan/test_data/protein.fasta \
  -m esm-scan/test_data/mutation_file.csv \
  -o esm-scan/results.csv
```

**Arguments:**
- `-i`: Input FASTA file.
- `-m`: (Optional) CSV file containing specific mutations (column: `mutant`). If omitted, a full scan will be performed.
- `-o`: Path to save the output CSV.

## 2. Check Endpoint Status

```bash
aws sagemaker describe-endpoint \
  --endpoint-name esm-scan-endpoint-async \
  --query "ProductionVariants[0].CurrentInstanceCount"
```

## 3. Forcefully Scale to Zero

```bash
aws sagemaker update-endpoint-weights-and-capacities \
  --endpoint-name esm-scan-endpoint-async \
  --desired-weights-and-capacities '[{"VariantName":"AllTraffic","DesiredInstanceCount":0}]'
```

Note: The endpoint scales to zero automatically after 15 minutes of inactivity.
