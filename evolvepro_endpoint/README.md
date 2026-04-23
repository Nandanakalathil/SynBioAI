

1. **Prerequisites**:
   - Python 3.8+
   - AWS CLI configured with appropriate permissions.

2. **Install Dependencies**:
   ```bash
   pip install boto3 pandas biopython requests
   ```

## Usage

### 1. Initial Master Scan (Zero-Shot)
Perform an initial zero-shot scan of your protein to generate master embeddings and first-round suggestions.

```bash
python invoke_evolvepro.py \
  --fasta protein.fasta \
```

### 2. Active Learning Round (Inference)
Suggest new mutants based on previous lab results (CSV) and the master embeddings generated in Round 1.

```bash
python invoke_evolvepro.py \
  --fasta protein.fasta \
  --labels labels.csv \
  --embeddings embeddings.csv \
  --measured_var activity \
  --max_mutations 1 \
  --num_mutants 16
```

## Parameters
- `--fasta`: Path to the wild-type FASTA file.
- `--labels`: (Optional) CSV containing lab results for mutation suggestions.
- `--embeddings`: (Optional) The master embeddings file generated in Round 1.
- `--measured_var`: Column name in the CSV representing the activity/fitness (default: `activity`).
- `--num_mutants`: Number of variants to suggest (default: 16).
- `--max_mutations`: Max mutations per variant (default: 1).
