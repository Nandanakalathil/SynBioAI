# ALDE 

This repository provides a fully automated pipeline for running **Active Learning-assisted Directed Evolution (ALDE)** experiments on AWS EC2. It implements the "Best Brain" configuration (AA Encodings + DKL Model + Thompson Sampling).

## Prerequisites

### 1. Local Environment
- **Python 3.7+**
- **AWS CLI**: Installed and configured (`aws configure`).
- **SSH Key**: An Ed25519 key at `~/.ssh/id_ed25519.pub`. If you don't have one, run:
  ```bash
  ssh-keygen -t ed25519
  ```
- **Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

## How to Run

Place your combinatorial landscape data in a CSV file (columns: `Combo`, `fitness`).

### Run on GB1 Dataset
```bash
python run_alde_experiment.py --input data/fitness.csv --output my_results.csv
```

##  Output
The script will generate a CSV file sorted by **fitness (descending)**. The variants at the top of this file represent the "Global Maximum" found by the ALDE algorithm—these are your leads for wet-lab synthesis.

---
