# DeepProzyme: Enzyme Commission (EC) Number Prediction

DeepProzyme (DeepEC) predicts the EC numbers for protein sequences to help annotate enzymatic functions.

## 1. Run Inference

```bash
python3 deepec/deepprozyme-inference.py \
  -i deepec/test_data/protein.fa \
  -o deepec/results.csv
```

**Arguments:**
- `-i`: Input FASTA file (one or more sequences).
- `-o`: Path to save the output CSV.

## 2. Check Endpoint Status

```bash
aws sagemaker describe-endpoint \
  --endpoint-name deepprozyme-endpoint \
  --query "ProductionVariants[0].CurrentInstanceCount"
```

## 3. Forcefully Scale to Zero

```bash
aws sagemaker update-endpoint-weights-and-capacities \
  --endpoint-name deepprozyme-endpoint \
  --desired-weights-and-capacities '[{"VariantName":"AllTraffic","DesiredInstanceCount":0}]'
```

Note: The endpoint scales to zero automatically after 15 minutes of inactivity.