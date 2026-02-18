#!/usr/bin/env python3
"""
S3F Async Inference CLI
Predict protein fitness using S3F (Sequence-Structure-Surface) model on SageMaker Async endpoint.
Requires: sequence (FASTA), structure (PDB), and optionally specific mutations (CSV).
"""

import argparse
import boto3
import json
import sys
import os
import csv
import time
import uuid
from botocore.exceptions import ClientError

# ================= CONFIG =================
ENDPOINT_NAME = "s3f-endpoint-async-v3"
S3_BUCKET = "synbioai-storage"
S3_PREFIX = "async-inference/s3f"
REGION = "us-east-1"
# ==========================================


def safe_exit(msg):
    print(f"\nERROR: {msg}")
    sys.exit(1)


def upload_to_s3(s3_client, payload, bucket, key):
    try:
        s3_client.put_object(Body=json.dumps(payload), Bucket=bucket, Key=key)
        return f"s3://{bucket}/{key}"
    except ClientError as e:
        safe_exit(f"S3 upload failed → {e}")


def poll_s3_result(s3_client, bucket, key, timeout=3600):
    """Poll S3 for inference result or error"""
    start_time = time.time()
    error_key = key + ".err"
    print(f"Waiting for results in s3://{bucket}/{key}...")

    while time.time() - start_time < timeout:
        # 1. Check for success
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            result_content = response["Body"].read().decode("utf-8")
            return json.loads(result_content)
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchKey":
                safe_exit(f"S3 polling error (Success Check) → {e}")

        # 2. Check for error file (.err)
        try:
            response = s3_client.get_object(Bucket=bucket, Key=error_key)
            error_content = response["Body"].read().decode("utf-8")
            safe_exit(f"Model Inference Failed! Error: {error_content}")
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchKey":
                safe_exit(f"S3 polling error (Error Check) → {e}")

        # 3. Wait and print status
        time.sleep(15)
        elapsed = int(time.time() - start_time)
        if elapsed % 60 == 0:
            print(
                f"  ...still waiting ({elapsed//60}m elapsed). "
                "If this is the first request, it may take 5-10 minutes to spin up the GPU."
            )

    safe_exit(f"Inference timed out after {timeout} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="S3F: Protein Fitness Prediction (Async)"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input FASTA file with protein sequence"
    )
    parser.add_argument(
        "-p", "--pdb", required=True, help="Input PDB file with protein structure"
    )
    parser.add_argument(
        "-m", "--mutations", help="Optional mutations CSV file (column: mutant)"
    )
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument(
        "--pdb-range",
        help="PDB range (e.g. '1-100'), default: auto-detect from structure",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Max wait time in seconds (default: 3600)",
    )

    args = parser.parse_args()

    # Start timing
    start_time_total = time.time()
    print(f"S3F inference started at: {time.ctime(start_time_total)}")

    # Validate input files
    if not os.path.exists(args.input):
        safe_exit(f"Input FASTA file not found: {args.input}")
    if not os.path.exists(args.pdb):
        safe_exit(f"PDB file not found: {args.pdb}")

    # 1. Parse FASTA
    fasta_sequence = ""
    protein_id = "protein"
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                protein_id = line[1:].split()[0]
            else:
                fasta_sequence += line

    if not fasta_sequence:
        safe_exit("No sequence found in FASTA file.")

    print(f"Protein: {protein_id}")
    print(f"Sequence length: {len(fasta_sequence)}")

    # 2. Read PDB file
    with open(args.pdb, "r") as f:
        pdb_content = f.read()

    if not pdb_content.strip():
        safe_exit("PDB file is empty.")
    print(f"PDB file: {args.pdb} ({len(pdb_content)} bytes)")

    # 3. Prepare payload
    payload = {
        "sequence": fasta_sequence,
        "pdb": pdb_content,
    }

    if args.pdb_range:
        payload["pdb_range"] = args.pdb_range

    # 4. Parse mutations if provided
    if args.mutations:
        if not os.path.exists(args.mutations):
            safe_exit(f"Mutations file not found: {args.mutations}")
        mutations = []
        with open(args.mutations, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                mut_idx = 0
                for i, col in enumerate(header):
                    if any(
                        x in col.lower()
                        for x in ["mutant", "mutation", "substitutions", "variant"]
                    ):
                        mut_idx = i
                        break
                for row in reader:
                    if row and len(row) > mut_idx:
                        mut = row[mut_idx].strip()
                        if mut:
                            mutations.append(mut)
        if mutations:
            payload["mutations"] = mutations
            print(f"Loaded {len(mutations)} specific mutations")
    else:
        total_possible = len(fasta_sequence) * 19  # 19 non-wt amino acids per position
        print(f"Full scan mode: will score all {total_possible} possible mutations")

    # 5. AWS clients
    try:
        s3 = boto3.client("s3", region_name=REGION)
        sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    except Exception as e:
        safe_exit(f"AWS client init failed → {e}")

    # 6. Upload to S3
    job_id = str(uuid.uuid4())
    input_key = f"{S3_PREFIX}/input/{job_id}.json"
    input_s3_uri = upload_to_s3(s3, payload, S3_BUCKET, input_key)
    print(f"Uploaded payload to {input_s3_uri}")

    # 7. Invoke endpoint
    print(f"Invoking {ENDPOINT_NAME} (Async)...")
    try:
        response = sm_runtime.invoke_endpoint_async(
            EndpointName=ENDPOINT_NAME,
            InputLocation=input_s3_uri,
            ContentType="application/json",
        )
        output_s3_uri = response["OutputLocation"]
        print(f"Output will be at: {output_s3_uri}")
    except ClientError as e:
        safe_exit(f"Endpoint invocation failed → {e}")

    # 8. Poll for result
    output_bucket = output_s3_uri.split("/")[2]
    output_key = "/".join(output_s3_uri.split("/")[3:])
    result_data = poll_s3_result(s3, output_bucket, output_key, timeout=args.timeout)

    # 9. Process results
    if "error" in result_data:
        safe_exit(f"Model returned error: {result_data['error']}")

    results = result_data.get("results", [])
    metadata = result_data.get("metadata", {})

    print(f"\n--- Results ---")
    print(f"Mutations scored: {metadata.get('total_mutations_scored', len(results))}")
    print(f"Server-side duration: {metadata.get('duration_seconds', '?')}s")

    # Save results
    print(f"Saving results to {args.output}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    count = 0
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["protein_id", "mutation", "score"])
        for item in results:
            if isinstance(item, dict):
                mutant = item.get("mutant")
                score = item.get("score")
                if score is not None:
                    writer.writerow([protein_id, mutant, score])
                    count += 1

    end_time_total = time.time()
    print(f"\nInference finished at: {time.ctime(end_time_total)}")
    print(f"Total duration: {int(end_time_total - start_time_total)} seconds")
    print(f"Saved {count} predictions to {args.output}")


if __name__ == "__main__":
    main()
