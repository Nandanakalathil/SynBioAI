"""
mpnn_client.py  —  Run ProteinMPNN from your local laptop via SageMaker Async Endpoint.

Install dependencies:
    pip install boto3 awscli

Configure AWS credentials (one-time):
    aws configure
    # Enter: Access Key, Secret Key, Region (us-east-1), output format (json)

Usage examples:
    # Basic — vanilla ProteinMPNN
    python mpnn_client.py --pdb my_protein.pdb --endpoint proteinmpnn-async --bucket my-protein-bucket

    # SolubleMPNN with more sequences
    python mpnn_client.py --pdb my_protein.pdb --endpoint proteinmpnn-async --bucket my-protein-bucket \
        --soluble --num-seq 5 --temp 0.2

    # Save output to file
    python mpnn_client.py --pdb my_protein.pdb --endpoint proteinmpnn-async --bucket my-protein-bucket \
        --output results.fasta
"""

import boto3
import json
import time
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ProteinMPNN client — calls SageMaker async endpoint")
parser.add_argument("--pdb",        required=True,               help="Path to local PDB file")
parser.add_argument("--endpoint",   required=True,               help="SageMaker endpoint name e.g. proteinmpnn-async")
parser.add_argument("--bucket",     required=True,               help="S3 bucket name e.g. my-protein-bucket")
parser.add_argument("--region",     default="us-east-1")
parser.add_argument("--soluble",    action="store_true",         help="Use SolubleMPNN weights")
parser.add_argument("--num-seq",    type=int,   default=2,       help="Number of sequences to generate")
parser.add_argument("--temp",       default="0.1",               help="Sampling temperature e.g. 0.1")
parser.add_argument("--seed",       type=int,   default=37)
parser.add_argument("--model-name", default="v_48_020")
parser.add_argument("--omit-aas",   default="X")
parser.add_argument("--output",     default="",                  help="Save sequences to this file")
parser.add_argument("--timeout",    type=int,   default=6000,     help="Max seconds to wait for result")
args = parser.parse_args()

# ── Clients ───────────────────────────────────────────────────────────────────
session         = boto3.Session(region_name=args.region)
s3              = session.client("s3")
sm_runtime      = session.client("sagemaker-runtime")

BUCKET          = args.bucket
S3_INPUT_PREFIX = "async-input"
S3_OUTPUT_PREFIX= "async-output"
ENDPOINT        = args.endpoint
PDB_PATH        = args.pdb

# ── Validate PDB ──────────────────────────────────────────────────────────────
if not os.path.exists(PDB_PATH):
    print(f"❌ PDB file not found: {PDB_PATH}")
    sys.exit(1)

pdb_name    = Path(PDB_PATH).stem
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id      = f"{pdb_name}_{timestamp}"

print(f"""
================================================
  ProteinMPNN Client
================================================
  PDB File    : {PDB_PATH}
  Endpoint    : {ENDPOINT}
  Model       : {"SolubleMPNN" if args.soluble else "ProteinMPNN"} ({args.model_name})
  Sequences   : {args.num_seq}
  Temperature : {args.temp}
  Job ID      : {job_id}
================================================
""")

# ── Step 1: Upload PDB to S3 ──────────────────────────────────────────────────
pdb_s3_key  = f"{S3_INPUT_PREFIX}/{job_id}/{pdb_name}.pdb"
payload_key = f"{S3_INPUT_PREFIX}/{job_id}/payload.json"

print(f"[1/4] Uploading PDB to s3://{BUCKET}/{pdb_s3_key}")
s3.upload_file(PDB_PATH, BUCKET, pdb_s3_key)

# ── Step 2: Build and upload payload ─────────────────────────────────────────
payload = {
    "pdb_path":           f"/tmp/{pdb_name}.pdb",   # path INSIDE container
    "use_soluble_model":  args.soluble,
    "num_seq_per_target": args.num_seq,
    "sampling_temp":      args.temp,
    "seed":               args.seed,
    "model_name":         args.model_name,
    "omit_AAs":           args.omit_aas,
    "backbone_noise":     0.0,
    # s3_pdb_uri is read by serve.py to download the PDB into the container
    "s3_pdb_uri":         f"s3://{BUCKET}/{pdb_s3_key}",
    "pdb_filename":       f"{pdb_name}.pdb",
}

payload_bytes = json.dumps(payload).encode()
print(f"[2/4] Uploading payload to s3://{BUCKET}/{payload_key}")
s3.put_object(Bucket=BUCKET, Key=payload_key, Body=payload_bytes, ContentType="application/json")

input_s3_uri    = f"s3://{BUCKET}/{payload_key}"
output_s3_uri   = f"s3://{BUCKET}/{S3_OUTPUT_PREFIX}/{job_id}/"

# ── Step 3: Invoke async endpoint ─────────────────────────────────────────────
print(f"[3/4] Invoking async endpoint: {ENDPOINT}")
response = sm_runtime.invoke_endpoint_async(
    EndpointName=ENDPOINT,
    InputLocation=input_s3_uri,
    ContentType="application/json",
    Accept="application/json",
)

output_location = response["OutputLocation"]
print(f"      Job submitted!")
print(f"      Output will appear at: {output_location}")

# ── Step 4: Poll for result ────────────────────────────────────────────────────
print(f"\n[4/4] Waiting for result (timeout={args.timeout}s)...", end="", flush=True)

# Parse bucket and key from output location
# Format: s3://bucket/key
output_parts    = output_location.replace("s3://", "").split("/", 1)
out_bucket      = output_parts[0]
out_key         = output_parts[1]

start_time  = time.time()
result      = None

while time.time() - start_time < args.timeout:
    try:
        obj     = s3.get_object(Bucket=out_bucket, Key=out_key)
        result  = json.loads(obj["Body"].read())
        print(f"\n      ✅ Result received in {int(time.time()-start_time)}s")
        break
    except s3.exceptions.NoSuchKey:
        # Also check for error file
        error_key = out_key + ".error"
        try:
            err_obj = s3.get_object(Bucket=out_bucket, Key=error_key)
            err     = err_obj["Body"].read().decode()
            print(f"\n      ❌ Inference error:\n{err}")
            sys.exit(1)
        except s3.exceptions.NoSuchKey:
            pass
        print(".", end="", flush=True)
        time.sleep(10)

if result is None:
    print(f"\n❌ Timeout after {args.timeout}s. Check S3 manually: {output_location}")
    sys.exit(1)

# ── Display results ────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  RESULTS")
print("="*50)

sequences = result.get("sequences", {})
if not sequences:
    print("⚠️  No sequences returned. stdout:")
    print(result.get("stdout", ""))
else:
    all_fasta = ""
    for fname, fasta in sequences.items():
        print(f"\n📄 {fname}:")
        print(fasta)
        all_fasta += fasta + "\n"

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write(all_fasta)
        print(f"\n💾 Sequences saved to: {args.output}")

print("\nDone! ✅")
