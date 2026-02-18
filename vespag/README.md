# VespaG: Variant Effect Scoring for Protein Sequences

VespaG is a fast and accurate model for scoring the impact of amino acid substitutions using genomic and protein sequence information. It supports both full-sequence scans and specific mutation scoring.

## 1. Run Inference

### Option A: Specific Mutations
```bash
python3 vespag/vespag-inference.py \
  -i vespag/test_data/protein.fasta \
  -m vespag/test_data/mutations.csv \
  -o vespag/results.csv \
  --normalize
```

### Option B: Full Sequence Scan
```bash
python3 vespag/vespag-inference.py \
  -i vespag/test_data/protein.fasta \
  -o vespag/results.csv \
  --normalize
```

**Arguments:**
- `-i`: Input FASTA file.
- `-m`: (Optional) CSV file containing specific mutations (column format: `protein_id,mutant`).
- `-o`: Path to save the output CSV.
- `--normalize`: (Optional) Normalizes the scores.

## 2. Check Endpoint Status

```bash
aws sagemaker describe-endpoint \
  --endpoint-name vespag-endpoint \
  --query "ProductionVariants[0].CurrentInstanceCount"
```

## 3. Forcefully Scale to Zero

```bash
aws sagemaker update-endpoint-weights-and-capacities \
  --endpoint-name vespag-endpoint \
  --desired-weights-and-capacities '[{"VariantName":"AllTraffic","DesiredInstanceCount":0}]'
```

Note: The endpoint scales to zero automatically after 15 minutes of inactivity.