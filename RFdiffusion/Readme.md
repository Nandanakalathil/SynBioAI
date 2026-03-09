# RFdiffusion: target-based protein and peptide design

RoseTTAFold diffusion (RFdiffusion) design binder protein or peptide directly from the target protein

## 1. Run Inference
This is for motif scaffolding:
```bash
python3 test_rfdiffusion.py \
  --pdb ./5TPN.pdb \
  --contigs "10-40/A163-181/10-40" \
  --num_designs 1
```
This is for binder design with hotspot:

```bash
python3 test_rfdiffusion.py \
  --pdb ./5TPN.pdb \
  --contigs "100-100/0 A20-35" \
  --num_designs 1 \
  --hotspot_res "A30,A33,A34"
```
**Arguments:**
- ` ./pdbs`: Input pdb file (one or more sequences).
  result will be saved in a folder in the same directory.

## 2. Check Endpoint Status

```bash
aws sagemaker describe-endpoint   --endpoint-name rfdiffusion-async-endpoint   --region us-east-1   --query "ProductionVariants[0].CurrentInstanceCount"
```

## 3. Forcefully Scale to Zero

```bash
aws sagemaker update-endpoint-weights-and-capacities \
--endpoint-name rfdiffusion-async-endpoint \
--desired-weights-and-capacities '[{"VariantName":"AllTraffic","DesiredInstanceCount":0}]'
```

Note: Scale the endpoint to zero automatically after using the inference.
