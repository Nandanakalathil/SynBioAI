# ProSST: Protein Structure and Sequence Transformer

ProSST is a model that leverages both sequence and structural information (via contact maps) to predict protein properties and variant effects.

## 1. Run Inference

```bash
python3 prosst/prosst-inference.py \
  -i prosst/test_data/protein.fasta \
  -p prosst/test_data/protein.pdb \
  -m prosst/test_data/mutations.csv \
  -o prosst/results.csv
```

**Arguments:**
- `-i`: Input FASTA file.
- `-p`: Input PDB file.
- `-m`: (Optional) CSV file containing mutations.
- `-o`: Path to save the output CSV.

## 2. Check Endpoint Status

```bash
aws sagemaker describe-endpoint \
  --endpoint-name prosst-inference-endpoint \
  --query "ProductionVariants[0].CurrentInstanceCount"
```

## 3. Forcefully Scale to Zero

```bash
aws sagemaker update-endpoint-weights-and-capacities \
  --endpoint-name prosst-inference-endpoint \
  --desired-weights-and-capacities '[{"VariantName":"AllTraffic","DesiredInstanceCount":0}]'
```

Note: The endpoint scales to zero automatically after 15 minutes of inactivity.