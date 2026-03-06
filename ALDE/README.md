
### 1. Initialize Project
Generate the design space and variant combinations.
```bash
./alde.sh generate-domain --name myproject --k 3
```

### 2. Initial Sampling
Pick the first batch of 96 random variants for testing.
```bash
./alde.sh initial-sample --name myproject --n_samples 96
```

### 3. Run Active Learning Rounds
Upload your lab results and get the next round of predictions in **one step**:
```bash
./alde.sh execute-round --name myproject --round 1 --file lab_results.csv
```

### 4. Cleanup
Wipe local and S3 project data to start fresh:
```bash
./alde.sh clean --name myproject
```

---

## 📂 Requirements
- **AWS CLI** installed and configured.
- That's it! No local Python or Torch needed.

## 📋 Project Structure (Automatic)
The script will automatically create these folders when you run commands:
- `PROJECT_NAME/data/`: Design space and training files.
- `PROJECT_NAME/results/`: AI-suggested variants for each round.
