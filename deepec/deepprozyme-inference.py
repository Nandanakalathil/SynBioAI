#!/usr/bin/env python3


import argparse
import boto3
import json
import sys
import os
import time
import uuid
from botocore.exceptions import ClientError

ENDPOINT_NAME = "deepprozyme-endpoint"
S3_BUCKET = "synbioai-storage"
S3_PREFIX = "async-inference/deepprozyme"
REGION = "us-east-1"

def upload_to_s3(s3_client, content, bucket, key):
    """Upload content to S3"""
    try:
        s3_client.put_object(Body=content, Bucket=bucket, Key=key)
        return f"s3://{bucket}/{key}"
    except ClientError as e:
        print(f"ERROR: S3 upload failed: {e}")
        sys.exit(1)

def poll_s3_result(s3_client, bucket, key, timeout=600):
    """Poll S3 for inference result"""
    start_time = time.time()
    print(f"Waiting for results in s3://{bucket}/{key}...")
    
    while time.time() - start_time < timeout:
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            result_content = response['Body'].read().decode('utf-8')
            return json.loads(result_content)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                # Check if endpoint is still scaling up or processing
                time.sleep(10)
                elapsed = int(time.time() - start_time)
                if elapsed % 30 == 0:
                    print(f"  ...still waiting ({elapsed}s elapsed)")
                continue
            else:
                print(f"ERROR: S3 polling error: {e}")
                sys.exit(1)
                
    print(f"ERROR: Inference timed out after {timeout} seconds.")
    sys.exit(1)

def invoke_async_endpoint(sm_runtime, input_s3_uri):
    """Invoke SageMaker async endpoint"""
    print(f"Invoking {ENDPOINT_NAME} (Async)...")
    try:
        response = sm_runtime.invoke_endpoint_async(
            EndpointName=ENDPOINT_NAME,
            InputLocation=input_s3_uri,
            ContentType="text/plain"
        )
        return response['OutputLocation']
    except ClientError as e:
        print(f"ERROR: Endpoint invocation failed: {e}")
        sys.exit(1)

def format_results(data, output_path):
    """Format results to TSV"""
    print(f"Formatting results to {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("sequence_ID\tprediction\tscore\n")
        
        for item in data:
            # Cleaner ID extraction
            full_id = item.get('id', 'unknown')
            pid = full_id.split()[0] if full_id else 'unknown'
            predictions = item.get('predictions', [])
            
            if not predictions:
                f.write(f"{pid}\tNone\t0.0\n")
            else:
                for p in predictions:
                    ec = p.get('ec', '')
                    score = p.get('score', 0.0)
                    f.write(f"{pid}\t{ec}\t{score}\n")
                    
    print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="DeepProzyme EC Number Prediction (Async)")
    parser.add_argument('-i', '--input', required=True, help='Input FASTA file')
    parser.add_argument('-o', '--output', required=True, help='Output file (TSV format)')
    parser.add_argument('--timeout', type=int, default=600, help='Max wait time in seconds (default: 600)')
    
    args = parser.parse_args()
    
    # Start timing
    start_time_total = time.time()
    print(f"Inference started at: {time.ctime(start_time_total)}")

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
        
    with open(args.input, 'r') as f:
        fasta_content = f.read()
    
    # Initialize AWS clients
    s3 = boto3.client('s3', region_name=REGION)
    sm_runtime = boto3.client('sagemaker-runtime', region_name=REGION)
    
    # 1. Upload input to S3
    job_id = str(uuid.uuid4())
    input_key = f"{S3_PREFIX}/input/{job_id}.fasta"
    input_s3_uri = upload_to_s3(s3, fasta_content, S3_BUCKET, input_key)
    
    # 2. Invoke endpoint
    output_s3_uri = invoke_async_endpoint(sm_runtime, input_s3_uri)
    
    # 3. Poll for result
    # OutputLocation is usually s3://bucket/prefix/output/uuid.out
    output_bucket = output_s3_uri.split('/')[2]
    output_key = '/'.join(output_s3_uri.split('/')[3:])
    
    result = poll_s3_result(s3, output_bucket, output_key, timeout=args.timeout)
    
    # 4. Format and save results
    format_results(result, args.output)

    end_time_total = time.time()
    print(f"Inference finished at: {time.ctime(end_time_total)}")
    print(f"Total duration: {int(end_time_total - start_time_total)} seconds")
    print("\nDeepProzyme inference complete!")

if __name__ == "__main__":
    main()
