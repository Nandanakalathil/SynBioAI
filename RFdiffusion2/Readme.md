# Omegafold: PLDDT score prediction in batch

Omegafold can predict plddt scores of protein sequences in batch.

## 1. Run Inference

```bash
python3 invoke_async.py --input test.csv --output omegafold_results.zip
```

**Arguments:**
- `-input`: Input csv file (one or more sequences).
- `-output`: result saved in zip file.

## 2. Check Endpoint Status

```bash
aws sagemaker describe-endpoint   --endpoint-name omegafold-async-endpoint   --region us-east-1   --query "ProductionVariants[0].CurrentInstanceCount"
```

## 3. Forcefully Scale to Zero

```bash
aws sagemaker update-endpoint-weights-and-capacities \
--endpoint-name omegafold-async-endpoint \
--desired-weights-and-capacities '[{"VariantName":"AllTraffic","DesiredInstanceCount":0}]'
```

Note: The endpoint scales to zero automatically after 15 minutes of inactivity.
