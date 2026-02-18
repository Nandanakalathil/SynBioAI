#!/usr/bin/env python3
"""
VespaG Async Inference CLI
Portable single-site variant effect scoring using SageMaker Asynchronous endpoint.
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
ENDPOINT_NAME = "vespag-endpoint"
S3_BUCKET = "synbioai-storage"
S3_PREFIX = "async-inference/vespag"
REGION = "us-east-1"
# ==========================================

def safe_exit(msg):
    print(f"\nERROR: {msg}")
    sys.exit(1)

def parse_fasta(path):
    if not os.path.exists(path):
        safe_exit(f"Input FASTA not found → {path}")

    sequences = {}
    current_id, current_seq = None, []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id:
            sequences[current_id] = "".join(current_seq)

    if not sequences:
        safe_exit("No sequences found in FASTA")

    return sequences

def parse_mutations(path):
    if not path:
        return None

    if not os.path.exists(path):
        safe_exit(f"Mutations file not found → {path}")

    mutations = []

    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and "protein" in line.lower():
                continue

            parts = line.strip().split(",")
            if len(parts) >= 2:
                mutations.append(f"{parts[0]}:{parts[1]}")

    return mutations if mutations else None

def save_results(predictions, output_path):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["protein_id", "mutation", "score"])

        for pid, scores in predictions.items():
            for mut, score in sorted(scores.items()):
                writer.writerow([pid, mut, score])
                count += 1

    print(f"Saved → {output_path}  ({count} predictions)")

def upload_to_s3(s3_client, payload, bucket, key):
    try:
        s3_client.put_object(Body=json.dumps(payload), Bucket=bucket, Key=key)
        return f"s3://{bucket}/{key}"
    except ClientError as e:
        safe_exit(f"S3 upload failed → {e}")

def poll_s3_result(s3_client, bucket, key, timeout=900):
    start_time = time.time()
    print(f"Waiting for results in s3://{bucket}/{key}...")
    
    while time.time() - start_time < timeout:
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            result_content = response['Body'].read().decode('utf-8')
            return json.loads(result_content)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                time.sleep(10)
                elapsed = int(time.time() - start_time)
                if elapsed % 30 == 0:
                    print(f"  ...still waiting ({elapsed}s elapsed)")
                continue
            else:
                safe_exit(f"S3 polling error → {e}")
                
    safe_exit(f"Inference timed out after {timeout} seconds")

def main():
    parser = argparse.ArgumentParser(description="VespaG Variant Effect Scoring (Async)")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA")
    parser.add_argument("-m", "--mutations", help="Optional mutation CSV")
    parser.add_argument("-o", "--output", required=True, help="Output CSV")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--timeout", type=int, default=900, help="Wait timeout in seconds")

    args = parser.parse_args()

    # Start timing
    start_time_total = time.time()
    print(f"Inference started at: {time.ctime(start_time_total)}")

    # 1. Parse input
    sequences = parse_fasta(args.input)
    mutations = parse_mutations(args.mutations)

    payload = {"sequences": sequences, "normalize": args.normalize}
    if mutations:
        payload["mutations"] = mutations

    # 2. AWS clients
    try:
        s3 = boto3.client("s3", region_name=REGION)
        sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    except Exception as e:
        safe_exit(f"AWS client init failed → {e}")

    # 3. Upload to S3
    job_id = str(uuid.uuid4())
    input_key = f"{S3_PREFIX}/input/{job_id}.json"
    input_s3_uri = upload_to_s3(s3, payload, S3_BUCKET, input_key)

    # 4. Invoke endpoint (Async)
    print(f"Invoking {ENDPOINT_NAME} (Async)...")
    try:
        response = sm_runtime.invoke_endpoint_async(
            EndpointName=ENDPOINT_NAME,
            InputLocation=input_s3_uri,
            ContentType="application/json"
        )
        output_s3_uri = response['OutputLocation']
    except ClientError as e:
        safe_exit(f"Endpoint invocation failed → {e}")

    # 5. Poll for result
    output_bucket = output_s3_uri.split('/')[2]
    output_key = '/'.join(output_s3_uri.split('/')[3:])
    
    result = poll_s3_result(s3, output_bucket, output_key, timeout=args.timeout)
    
    predictions = result.get("predictions", {})
    if not predictions:
        safe_exit("No predictions returned from endpoint")

    # 6. Save results
    save_results(predictions, args.output)

    end_time_total = time.time()
    print(f"Inference finished at: {time.ctime(end_time_total)}")
    print(f"Total duration: {int(end_time_total - start_time_total)} seconds")
    print("\nVespaG inference complete!")

if __name__ == "__main__":
    main()
