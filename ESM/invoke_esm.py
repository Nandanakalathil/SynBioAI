import argparse
import pandas as pd
import requests
import io
import os
import boto3
from Bio import SeqIO

def chunk_sequences(fasta_file, batch_size):
    """Chunk FASTA sequences into batches."""
    records = list(SeqIO.parse(fasta_file, "fasta"))
    for i in range(0, len(records), batch_size):
        yield records[i:i + batch_size]

def invoke_local(fasta_content, url="http://localhost:8080/invocations"):
    """Invoke local Flask server."""
    response = requests.post(url, data=fasta_content, headers={"Content-Type": "text/plain"})
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text), index_col=0)
    else:
        print(f"Error ({response.status_code}): {response.text}")
        return None

def invoke_sagemaker(fasta_content, endpoint_name):
    """Invoke SageMaker endpoint (synchronous)."""
    runtime = boto3.client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/plain",
        Body=fasta_content
    )
    csv_str = response["Body"].read().decode("utf-8")
    return pd.read_csv(io.StringIO(csv_str), index_col=0)

def invoke_sagemaker_async(fasta_content, endpoint_name, bucket, prefix):
    """Invoke SageMaker Asynchronous endpoint."""
    s3 = boto3.client("s3")
    runtime = boto3.client("sagemaker-runtime")
    
    # Generate unique ID for this request
    req_id = str(uuid.uuid4())[:8]
    input_key = f"{prefix}/inputs/{req_id}.fasta"
    
    print(f"[*] Uploading input to s3://{bucket}/{input_key}...")
    s3.put_object(Bucket=bucket, Key=input_key, Body=fasta_content)
    
    response = runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=f"s3://{bucket}/{input_key}",
        ContentType="text/plain"
    )
    
    output_location = response["OutputLocation"]
    print(f"[*] Async request submitted. Output will be at: {output_location}")
    print("[*] Polling for results...")
    
    # Polling logic
    b, k = output_location.replace("s3://", "").split("/", 1)
    max_wait = 600  # 10 minutes
    waited = 0
    while waited < max_wait:
        try:
            resp = s3.get_object(Bucket=b, Key=k)
            csv_str = resp["Body"].read().decode("utf-8")
            print(f"[+] Result received after {waited}s")
            return pd.read_csv(io.StringIO(csv_str), index_col=0)
        except s3.exceptions.NoSuchKey:
            time.sleep(10)
            waited += 10
            if waited % 30 == 0:
                print(f"    ... waiting for result ({waited}s)")
    
    print("[!] Timeout waiting for async result.")
    return None

import uuid
import time

def main():
    parser = argparse.ArgumentParser(description="Invoke ESM2 Embedding Endpoint")
    parser.add_argument("--fasta", type=str, required=True, help="Path to input FASTA file")
    parser.add_argument("--output", type=str, default="embeddings.csv", help="Path to output CSV file")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of sequences per batch request")
    parser.add_argument("--endpoint_name", type=str, help="SageMaker endpoint name")
    parser.add_argument("--local_url", type=str, default="http://localhost:8080/invocations", help="Local server URL")
    parser.add_argument("--is_async", action="store_true", help="Use asynchronous invocation")
    parser.add_argument("--s3_bucket", type=str, help="S3 bucket for async input/output")
    parser.add_argument("--s3_prefix", type=str, default="esm2/async-io", help="S3 prefix for async data")

    args = parser.parse_args()

    if args.is_async and not args.endpoint_name:
        print("[!] --endpoint_name is required for async invocation.")
        return

    if args.is_async and not args.s3_bucket:
        import sagemaker
        args.s3_bucket = sagemaker.Session().default_bucket()
        print(f"[*] Using default S3 bucket: {args.s3_bucket}")

    all_dfs = []
    
    print(f"[*] Processing {args.fasta} with batch size {args.batch_size}...")
    
    for chunk in chunk_sequences(args.fasta, args.batch_size):
        # Create temporary FASTA content for the batch
        fasta_io = io.StringIO()
        SeqIO.write(chunk, fasta_io, "fasta")
        fasta_content = fasta_io.getvalue()

        if args.endpoint_name:
            if args.is_async:
                print(f"[*] Invoking SageMaker ASYNC endpoint '{args.endpoint_name}' for {len(chunk)} sequences...")
                df = invoke_sagemaker_async(fasta_content, args.endpoint_name, args.s3_bucket, args.s3_prefix)
            else:
                print(f"[*] Invoking SageMaker endpoint '{args.endpoint_name}' for {len(chunk)} sequences...")
                df = invoke_sagemaker(fasta_content, args.endpoint_name)
        else:
            print(f"[*] Invoking local server for {len(chunk)} sequences...")
            df = invoke_local(fasta_content, args.local_url)
        
        if df is not None:
            all_dfs.append(df)
            print(f"[*] Received embeddings for {len(df)} sequences")

    if all_dfs:
        final_df = pd.concat(all_dfs)
        final_df.to_csv(args.output)
        print(f"[+] Total embeddings saved to {args.output} (Shape: {final_df.shape})")
    else:
        print("[!] No embeddings generated.")

if __name__ == "__main__":
    main()
