# ALDE SageMaker Client

## Overview
This folder contains the CLI and helper tools to interact with the **ALDE SageMaker Asynchronous Endpoint**. It uses a **DNN Ensemble + One-Hot Encoding + Thompson Sampling (TS)** to iteratively discover high-fitness protein variants.

---

## Workflow Guide

### 1. Project Initialization
Generate the design space and pick your first random batch of variants.
```bash
# 1. Generate all possible combinations
python alde_cli.py generate-domain --name myproject --k 3

# 2. Pick 96 random variants for the first experiment
python alde_cli.py initial-sample --name myproject --n_samples 96
```
The file `initial_samples_to_test.csv` will appear in `myproject/data/`.

### 2. Prepare Training Data (Manual)
Once you have lab results for the initial 96, create a file named `init.csv` in your project folder with two columns: `Combo` and `Fitness`.
*Note: Ensure NO spaces after commas and standard line endings.*

### 3. Run Active Learning Rounds
This is the core loop. Each round takes your current data and proposes 96 new variants.

**A. Upload your latest data:**
```bash
python alde_cli.py upload --name myproject --file myproject/data/init.csv
```

**B. Execute the Round:**
```bash
python alde_cli.py execute-round --name myproject --round 1 --data_csv init.csv
```

**C. Add Results & Merge:**
Open `myproject/results/round1/round1_proposals.csv`. It contains just the 96 sequences. Add a `Fitness` column and fill in your lab results.

Then, merge it to create the training file for the next round:
```bash
conda run -n alde python alde_helpers.py merge \
  --name myproject \
  --current myproject/data/myproject/init.csv \
  --proposals myproject/results/round1/round1_proposals.csv \
  --output myproject/data/myproject/r1.csv
```

---
## Commands Reference

### `alde_cli.py` (Cloud Interaction)
| Command | Description |
| `generate-domain` | Creates the `all_combos.csv` and `onehot_x.pt` on S3. |
| `initial-sample` | Picks the first 96 random variants. |
| `upload` | Sends your local CSV (with lab results) to S3. |
| `execute-round` | The main heart of the AI. Trains the model and picks new variants. |
| `sync` | Manually pull the latest files from S3 to your Mac. |

### `alde_helpers.py` (Local Processing)
| Command | Description |
| :--- | :--- |
| `extract` | Converts raw AI output into a readable `roundX_proposals.csv`. (Now called automatically by `execute-round`). |
| `merge` | Combines your old training data with new lab results to make a bigger training set for the next round. |

---

## Tips & Troubleshooting
- **Scale-to-Zero**: The endpoint automatically shuts down after 15 minutes of inactivity to save costs. The CLI will wake it up next time you run a command.
