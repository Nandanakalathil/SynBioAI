# ProteinMPNN: Redesign protein sequences

ProteinMPNN can redesign sequences and give high plddt, ptm, iptm score sequences.

## 1. Run Inference

```bash
## ProteinMPNN:
python3 mpnn_client.py --pdb protein.pdb \
    --endpoint proteinmpnn-async \
    --bucket sagemaker-us-east-1-119492371915 \
    --num-seq 10 \
    --temp 0.2 \
    --output protein_results.fasta
```
```bash
## solubleMPNN:
python3 mpnn_client.py --pdb protein.pdb \
    --endpoint proteinmpnn-async \
    --bucket sagemaker-us-east-1-119492371915 \
    --soluble \
    --num-seq 5 \
    --output protein_soluble_results.fasta
```

**Arguments:**
- `-pdb`: Input pdb file (one or more sequences).
- `-output`: result saved in fasta file.

## 2. Check Endpoint Status

```bash
aws sagemaker describe-endpoint \
    --endpoint-name proteinmpnn-async \
    --region us-east-1 \
    --query "[EndpointStatus,ProductionVariants[0].CurrentInstanceCount]"
```

## 3. Forcefully Scale to Zero

aws application-autoscaling register-scalable-target \
    --service-namespace sagemaker \
    --resource-id endpoint/proteinmpnn-async/variant/AllTraffic \
    --scalable-dimension sagemaker:variant:DesiredInstanceCount \
    --min-capacity 0 \
    --max-capacity 0 \
    --region us-east-1

## 3. Forcefully Scale to one

aws application-autoscaling register-scalable-target \
    --service-namespace sagemaker \
    --resource-id endpoint/proteinmpnn-async/variant/AllTraffic \
    --scalable-dimension sagemaker:variant:DesiredInstanceCount \
    --min-capacity 0 \
    --max-capacity 1 \
    --region us-east-1

Note: The endpoint scales to zero automatically after 15 minutes of inactivity.
