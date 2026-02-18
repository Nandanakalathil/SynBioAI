# S3F: Sequence-Structure-Surface Fitness Model

S3F is a comprehensive model for predicting protein fitness by integrating protein sequence (ESM-2), structure (GVP), and surface features (PyKeOps).

## 1. Run Inference

```bash
python3 s3f/s3f-inference.py \
  -i s3f/test_data/protein.fasta \
  -p s3f/test_data/protein.pdb \
  -m s3f/test_data/mutations.csv \
  -o s3f/test_data/result.csv
```

**Arguments:**
- `-i`: Input FASTA file.
- `-p`: Input PDB file.
- `-m`: (Optional) CSV file containing specific mutations (column: `mutant`).
- `-o`: Path to save the output CSV.

## 2. Check Endpoint Status

```bash
aws sagemaker describe-endpoint \
  --endpoint-name s3f-endpoint-async-v3 \
  --query "ProductionVariants[0].CurrentInstanceCount"
```

## 3. Forcefully Scale to Zero

```bash
aws sagemaker update-endpoint-weights-and-capacities \
  --endpoint-name s3f-endpoint-async-v3 \
  --desired-weights-and-capacities '[{"VariantName":"AllTraffic","DesiredInstanceCount":0}]'
```

Note: The first request after scaling to zero will take 5-8 minutes to spin up the GPU instance.
