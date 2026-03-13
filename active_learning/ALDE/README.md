# ALDE: Active Learning-assisted Directed Evolution

**Active Learning-assisted Directed Evolution (ALDE)** is a modeling and simulation package designed to perform active learning on combinatorial site-saturation libraries. This robust pipeline is built to systematically accelerate protein engineering.

By default, this repository is configured to use our highest-performing active learning pipeline:
- **Encoding:** ESM-2 (Evolutionary Scale Modeling)
- **Model:** Deep Kernel Learning (`DKL_BOTORCH`) 
- **Acquisition Function:** Thompson Sampling (`TS`)

---

##  Installation

Clone this repository and create the required Anaconda environment:

```bash
# Build and activate the environment
conda env create -f alde.yml
conda activate alde

# (Required to support ESM-2 protein encoding)
pip install fair-esm
```

> **Data Download (Optional):** The pre-computed encodings and fitness data from our study can be downloaded [here](https://zenodo.org/records/12196802). Unzip it and replace the empty `data` folder. If you are using your own sequences and data, you can skip this step and generate them yourself using the commands below.

---

## Production Runs (Wet-Lab Campaign)

Production runs are used to execute a standard wet-lab ALDE campaign, predicting the most optimal sequences to synthesize in your next lab batch. 

### 1. Generate the Design Space
First, you must create the foundational dataset containing all possible protein variants and their deep-learning encodings. This step only needs to be run **once** per campaign.

Run `generate_domain.py` while specifying the number of mutable residues (`nsites`):

```bash
python generate_domain.py --name=ParPgb --nsites=5 --encoding=ESM2
```

**Outputs** (saved in `data/{name}/`):
* `all_combos.csv`: The list of all generated sequences (the "domain").
* `ESM2_x.pt`: A torch tensor containing the high-dimensional ESM-2 embeddings for every single sequence. *(Note: depending on the size of your library, evaluating ESM-2 can be computationally heavy. A GPU is recommended).*

### 2. Add Training Data
Before the model can predict, it requires some initial truth. Upload your lab results into the `data/{name}/` folder as a CSV (e.g., `fitness_round1_training.csv`).

**Must contain:**
* `Combo`: The protein sequence string.
* A numeric column representing the fitness value to optimize (e.g., `Diff`).

### 3. Predict the Next Batch
Execute the ALDE pipeline to train the `DKL_BOTORCH` model on your current wet-lab data, evaluating uncertainty using `Thompson Sampling` to propose the next optimal variants to test:

```bash
python execute_production.py \
    --name=ParPgb \
    --data_csv=fitness_round1_training.csv \
    --obj_col=Diff \
    --output_path=results/ParPgb_production/round1/ \
    --batch_size=96
```
