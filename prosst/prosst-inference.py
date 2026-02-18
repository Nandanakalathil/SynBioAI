#!/usr/bin/env python3
"""
ProSST Async Inference CLI
Structure-aware protein variant scoring using ProSST on SageMaker Asynchronous endpoint.
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
ENDPOINT_NAME = "prosst-inference-endpoint"
S3_BUCKET = "synbioai-storage"
S3_PREFIX = "async-inference/prosst"
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
            result_content = response['Body'].read().decode('utf-8')
            return json.loads(result_content)
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                safe_exit(f"S3 polling error (Success Check) → {e}")
        
        # 2. Check for error file (.err)
        try:
            response = s3_client.get_object(Bucket=bucket, Key=error_key)
            error_content = response['Body'].read().decode('utf-8')
            safe_exit(f"Model Inference Failed! Error: {error_content}")
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                safe_exit(f"S3 polling error (Error Check) → {e}")

        # 3. Wait and print status
        time.sleep(15)
        elapsed = int(time.time() - start_time)
        if elapsed % 60 == 0:
            print(f"  ...still waiting ({elapsed//60}m elapsed). If this is the first request, it may take 5-8 minutes to spin up the GPU.")
                
    safe_exit(f"Inference timed out after {timeout} seconds")

def main():
    parser = argparse.ArgumentParser(description="ProSST Structure-Aware Protein Scoring (Async)")
    parser.add_argument('-i', '--input', required=True, help='Input FASTA file')
    parser.add_argument('-p', '--pdb', required=True, help='PDB structure file')
    parser.add_argument('-m', '--mutations', help='Optional mutations CSV file')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file')
    parser.add_argument('--timeout', type=int, default=1200, help='Max wait time in seconds (default: 1200)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input): safe_exit(f"Input file not found: {args.input}")
    if not os.path.exists(args.pdb): safe_exit(f"PDB file not found: {args.pdb}")
        
    # 1. Parse PDB and FASTA
    print(f"Reading PDB structure from {args.pdb}...")
    res_names = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    pdb_sequence = []
    try:
        with open(args.pdb, 'r') as f:
            pdb_content = f.read()
            f.seek(0)
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    res_name = line[17:20].strip()
                    aa = res_names.get(res_name, 'X')
                    pdb_sequence.append(aa)
    except Exception as e:
        safe_exit(f"Failed to read PDB file: {e}")
    
    sequence_from_pdb = "".join(pdb_sequence)
    if not sequence_from_pdb:
        print("Warning: Could not extract sequence from PDB file (no CA atoms found).")
    
    fasta_sequence = ""
    protein_id = "protein"
    try:
        with open(args.input, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    protein_id = line[1:].split()[0]
                else:
                    fasta_sequence += line
    except Exception as e:
        safe_exit(f"Failed to read FASTA file: {e}")
    
    sequence = sequence_from_pdb if sequence_from_pdb else fasta_sequence
    if not sequence:
        safe_exit("No sequence found in either FASTA or PDB file.")

    # 2. Prepare payload and parse mutations
    if not args.mutations:
        safe_exit("ProSST requires a mutations file (-m) to identify variants for scoring. Please provide a CSV file with a 'mutant' column (e.g., A10G).")

    if not os.path.exists(args.mutations):
        safe_exit(f"Mutations file not found: {args.mutations}")

    mutations = []
    print(f"Reading mutations from {args.mutations}...")
    try:
        with open(args.mutations, 'r') as f:
            # Check if first line is a header
            first_line = f.readline()
            f.seek(0)
            has_header = any(x in first_line.lower() for x in ['mutant', 'mutation', 'substitutions', 'protein', 'id'])
            
            reader = csv.reader(f)
            mut_idx = 0
            
            if has_header:
                header = next(reader)
                for i, col in enumerate(header):
                    if any(x in col.lower() for x in ['mutant', 'mutation', 'substitutions', 'variant']):
                        mut_idx = i
                        break
            
            for row in reader:
                if row and len(row) > mut_idx:
                    mut = row[mut_idx].strip()
                    if mut: mutations.append(mut)
    except Exception as e:
        safe_exit(f"Failed to parse mutations CSV: {e}")

    if not mutations:
        safe_exit("No valid mutations found in the provided CSV file. Ensure the file contains a column named 'mutant' or 'mutation'.")

    print(f"Validated input: Sequence length {len(sequence)}, PDB content loaded, {len(mutations)} mutations found.")

    input_data = {
        "fasta": sequence,
        "pdb": pdb_content,
        "mutants": mutations
    }

    # 3. AWS clients
    try:
        s3 = boto3.client('s3', region_name=REGION)
        sm_runtime = boto3.client('sagemaker-runtime', region_name=REGION)
    except Exception as e: safe_exit(f"AWS client init failed → {e}")

    # 4. Upload to S3
    job_id = str(uuid.uuid4())
    input_key = f"{S3_PREFIX}/input/{job_id}.json"
    input_s3_uri = upload_to_s3(s3, input_data, S3_BUCKET, input_key)

    # 5. Invoke endpoint
    print(f"Invoking {ENDPOINT_NAME} (Async)...")
    try:
        response = sm_runtime.invoke_endpoint_async(
            EndpointName=ENDPOINT_NAME,
            InputLocation=input_s3_uri,
            ContentType="application/json"
        )
        output_s3_uri = response['OutputLocation']
    except ClientError as e: safe_exit(f"Endpoint invocation failed → {e}")

    # 6. Poll for result
    output_bucket = output_s3_uri.split('/')[2]
    output_key = '/'.join(output_s3_uri.split('/')[3:])
    result_data = poll_s3_result(s3, output_bucket, output_key, timeout=args.timeout)
    
    # 7. Handle Errors in Response
    if "error" in result_data:
        safe_exit(f"Model returned error: {result_data['error']}")

    # 8. Process results
    results = result_data.get('results', [])
    if not results and not isinstance(results, list):
         safe_exit(f"Unexpected response format: {result_data}")

    print(f"Saving results to {args.output}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    count = 0
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['protein_id', 'mutation', 'score'])
        
        for item in results:
            if isinstance(item, dict):
                mutant = item.get('mutant')
                score = item.get('score')
                if score is not None:
                    writer.writerow([protein_id, mutant, score])
                    count += 1
            elif isinstance(item, list) and len(item) >= 2:
                # Fallback for alternative formats
                writer.writerow([protein_id, item[0], item[1]])
                count += 1
    
    if count == 0:
        print("Warning: No valid predictions were found in the results.")
    else:
        print(f"Inference complete! Saved {count} predictions.")

if __name__ == "__main__":
    main()
