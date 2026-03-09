# RFdiffusion2: Atom-level enzyme active site scaffolding using RFdiffusion2

RoseTTAFold diffusion 2 (RFdiffusion2) design enzymes directly from functional group geometries without specifying residue order or performing inverse rotamer generation

## 1. Run Inference

```bash
python3 invoke_endpoint.py \
    --local-pdbs 1qys.pdb \
    --input-bucket filtarationmetrices \
    --output-bucket filtarationmetrices \
    --benchmark active_site_unindexed_atomic_partial_ligand \
    --num-designs 1 \
    --region us-east-1
```

**Arguments:**
- `-local-pdbs`: Input pdb file (one or more sequences).
  `output`: result will be saved in zip file.

## 2. Check Endpoint Status

```bash
aws sagemaker describe-endpoint   --endpoint-name rfdiffusion2-async-endpoint   --region us-east-1   --query "ProductionVariants[0].CurrentInstanceCount"
```

## 3. Forcefully Scale to Zero

```bash
aws sagemaker update-endpoint-weights-and-capacities \
--endpoint-name rfdiffusion2-async-endpoint \
--desired-weights-and-capacities '[{"VariantName":"AllTraffic","DesiredInstanceCount":0}]'
```

Note: The endpoint scales to zero automatically after 15 minutes of inactivity.
