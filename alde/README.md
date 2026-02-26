# ALDE: Active Learning for Directed Evolution
This directory contains the implementation and deployment scripts for **ALDE (Active Learning for Directed Evolution)**. ALDE recommends optimal protein sequences based on provided fitness data, efficiently guiding the search through large sequence spaces.

## Running Inference
Use the `alde_invoke.py` script to submit a CSV file and get recommendations. 

**Default Command:**
```bash
python3.10 alde_invoke.py --csv /Users/nandana/Desktop/SynBioAI_endpoints/alde/test_data/GB1.csv --output results/GB1_results.csv
```

**Arguments:**
- `--csv`: Path to your input CSV (must contain `Combo` and fitness columns).
- `--output`: Local path where the results will be saved.

##  Input Format
The input CSV should follow this structure:

| Combo | fitness |
| :--- | :--- |
| AAAA | 0.0744 |
| AAAC | 0.0563 |
| ... | ... |

Note: The `Combo` column represents the protein sequence string.
