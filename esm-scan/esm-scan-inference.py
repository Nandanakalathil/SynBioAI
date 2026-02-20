#!/usr/bin/env python3
"""
ESM-Scan Async Inference CLI
Predict the impact of mutations on a protein using ESM-Scan on SageMaker Async endpoint.
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
ENDPOINT_NAME = "esm-scan-endpoint-async"
S3_BUCKET = "synbioai-storage"
S3_PREFIX = "async-inference/esm-scan"
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


def poll_s3_result(s3_client, bucket, key, timeout=1800):
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
                "If this is the first request, it may take 5-8 minutes to spin up the GPU."
            )

    safe_exit(f"Inference timed out after {timeout} seconds")


def main():
    parser = argparse.ArgumentParser(description="ESM-Scan: Variant Effect Prediction (Async)")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file with protein sequence")
    parser.add_argument("-m", "--mutations", help="Optional mutations CSV file (column: mutant)")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument(
        "-s", "--scoring-strategy",
        default="wt-marginals",
        choices=["wt-marginals", "masked-marginals"],
        help="Scoring strategy (default: wt-marginals, faster)",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="Max wait time in seconds (default: 1800)")

    args = parser.parse_args()

    # Start timing
    start_time_total = time.time()
    print(f"ESM-Scan inference started at: {time.ctime(start_time_total)}")

    if not os.path.exists(args.input):
        safe_exit(f"Input file not found: {args.input}")

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

    # 2. Prepare payload
    payload = {
        "sequence": fasta_sequence,
        "scoring_strategy": args.scoring_strategy,
    }

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
            # --- Auto-fix: Detect 0-based indexing ---
            matches_1based = 0
            matches_0based = 0
            mutations_parsed = []
            
            for m in mutations:
                if len(m) < 2: continue
                wt_char = m[0]
                try:
                    pos = int(m[1:-1])
                except ValueError:
                    continue # Skip invalid format
                
                # Check 1-based (standard)
                if 1 <= pos <= len(fasta_sequence):
                    if fasta_sequence[pos-1] == wt_char:
                        matches_1based += 1
                
                # Check 0-based (alternative)
                if 0 <= pos < len(fasta_sequence):
                    if fasta_sequence[pos] == wt_char:
                        matches_0based += 1
                
                mutations_parsed.append((m, wt_char, pos, m[-1]))

            if matches_0based > matches_1based and matches_0based > 0:
                print(f"WARNING: Detected 0-based indexing in mutation file (matches: 0-based={matches_0based}, 1-based={matches_1based}).")
                print("         Automatically adjusting mutations to 1-based indexing (pos + 1).")
                fixed_mutations = []
                for orig, wt, pos, mut in mutations_parsed:
                    new_pos = pos + 1
                    fixed_mutations.append(f"{wt}{new_pos}{mut}")
                mutations = fixed_mutations
            elif matches_1based == 0 and len(mutations) > 0:
                 print(f"WARNING: No mutations matched the sequence with standard 1-based indexing (WT mismatch).")
                 print(f"         Please verifying your sequence and mutation file numbering.")
            
            payload["mutations"] = mutations
            print(f"Loaded {len(mutations)} specific mutations")
    else:
        total_possible = len(fasta_sequence) * 20
        # Warn if full scan could be extremely large (e.g., > 200,000 mutations)
        if total_possible > 200000:
            print(f"WARNING: Full scan will evaluate {total_possible} mutations which may take a long time and cause timeout.")
            print("Consider providing a mutations CSV file to limit the search or increase the timeout.")
        print(f"Full scan mode: will score all {total_possible} possible mutations")

    # 3. AWS clients
    try:
        s3 = boto3.client("s3", region_name=REGION)
        sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
        sm = boto3.client("sagemaker", region_name=REGION)
    except Exception as e:
        safe_exit(f"AWS client init failed → {e}")

    # 4. Upload to S3
    job_id = str(uuid.uuid4())
    input_key = f"{S3_PREFIX}/input/{job_id}.json"
    input_s3_uri = upload_to_s3(s3, payload, S3_BUCKET, input_key)
    print(f"Uploaded payload to {input_s3_uri}")

    # 5. Invoke endpoint
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

    # 6. Poll for result
    output_bucket = output_s3_uri.split("/")[2]
    output_key = "/".join(output_s3_uri.split("/")[3:])
    result_data = poll_s3_result(s3, output_bucket, output_key, timeout=args.timeout)

    # 7. Process results
    if "error" in result_data:
        safe_exit(f"Model returned error: {result_data['error']}")

    results = result_data.get("results", [])
    metadata = result_data.get("metadata", {})

    print(f"\n--- Results ---")
    print(f"Mutations scored: {metadata.get('total_mutations_scored', len(results))}")
    print(f"Scoring strategy: {metadata.get('scoring_strategy', 'unknown')}")
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
