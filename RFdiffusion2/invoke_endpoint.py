"""
invoke_endpoint.py — Submit PDB files to RFdiffusion2 SageMaker Async Endpoint

Usage:
    # Single PDB
    python invoke_endpoint.py \
        --pdbs s3://your-bucket/inputs/protein1.pdb \
        --output-bucket your-bucket \
        --region us-east-1

    # Multiple PDBs
    python invoke_endpoint.py \
        --pdbs s3://your-bucket/inputs/protein1.pdb s3://your-bucket/inputs/protein2.pdb \
        --output-bucket your-bucket \
        --benchmark active_site_unindexed_atomic_partial_ligand \
        --num-designs 1 \
        --region us-east-1

    # Upload a local PDB file first, then run
    python invoke_endpoint.py \
        --local-pdbs /path/to/protein1.pdb /path/to/protein2.pdb \
        --input-bucket your-bucket \
        --output-bucket your-bucket \
        --region us-east-1
"""

import argparse
import json
import time
import os
import uuid
import boto3

ENDPOINT_NAME = "rfdiffusion2-async-endpoint"


def upload_local_pdb(local_path, bucket, prefix="rfdiffusion2/inputs"):
    """Upload a local PDB file to S3, return s3:// path."""
    s3     = boto3.client("s3")
    key    = f"{prefix}/{os.path.basename(local_path)}"
    s3_path = f"s3://{bucket}/{key}"
    print(f"Uploading {local_path} -> {s3_path}")
    s3.upload_file(local_path, bucket, key)
    return s3_path


def invoke_async_endpoint(pdb_s3_paths, output_bucket, parameters, region):
    """Submit inference request to the async endpoint."""
    sm_runtime = boto3.client("sagemaker-runtime", region_name=region)

    payload = {
        "pdbs":              pdb_s3_paths,
        "output_s3_bucket":  output_bucket,
        "output_s3_prefix":  "rfdiffusion2/outputs",
        "parameters":        parameters,
    }

    print(f"\nSubmitting {len(pdb_s3_paths)} PDB(s) to endpoint: {ENDPOINT_NAME}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    response = sm_runtime.invoke_endpoint_async(
        EndpointName    = ENDPOINT_NAME,
        InputLocation   = upload_input_to_s3(payload, output_bucket),
        ContentType     = "application/json",
    )

    output_location = response["OutputLocation"]
    print(f"\n✅ Job submitted!")
    print(f"   Output will be at: {output_location}")
    return output_location


def upload_input_to_s3(payload, bucket):
    """Upload the JSON payload to S3 (required for async endpoints)."""
    s3       = boto3.client("s3")
    job_id   = str(uuid.uuid4())[:8]
    key      = f"rfdiffusion2/async-inputs/{job_id}/input.json"
    s3_path  = f"s3://{bucket}/{key}"

    s3.put_object(
        Bucket      = bucket,
        Key         = key,
        Body        = json.dumps(payload),
        ContentType = "application/json",
    )
    print(f"Uploaded input payload to: {s3_path}")
    return s3_path


def poll_for_output(output_location, region, timeout_minutes=240):
    """Poll S3 until the async output is ready."""
    s3 = boto3.client("s3", region_name=region)

    # Parse s3://bucket/key
    path   = output_location.replace("s3://", "")
    bucket, key = path.split("/", 1)

    print(f"\nPolling for output at: {output_location}")
    print(f"Timeout: {timeout_minutes} minutes")

    elapsed = 0
    interval = 30  # seconds

    while elapsed < timeout_minutes * 60:
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            result   = json.loads(response["Body"].read())
            print(f"\n✅ Job complete!")
            print(json.dumps(result, indent=2))

            # Auto-download results to current directory
            print("\n--- Downloading results ---")
            for r in result.get("results", []):
                if r["status"] == "success":
                    s3_path  = r["output_s3"]
                    filename = s3_path.split("/")[-1]
                    zip_bucket, zip_key = s3_path.replace("s3://", "").split("/", 1)
                    local_zip = f"./{filename}"
                    print(f"Downloading {s3_path} -> {local_zip}")
                    s3.download_file(zip_bucket, zip_key, local_zip)
                    import zipfile, os
                    extract_dir = f"./{filename.replace('.zip', '')}"
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(local_zip, 'r') as zf:
                        zf.extractall(extract_dir)
                    print(f"Results saved to: {extract_dir}/")

            return result

        except s3.exceptions.NoSuchKey:
            print(f"  Waiting... ({elapsed}s elapsed)")
            time.sleep(interval)
            elapsed += interval

        except Exception as e:
            print(f"  Error polling: {e}")
            time.sleep(interval)
            elapsed += interval

    print(f"⏰ Timeout after {timeout_minutes} minutes")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdbs",       nargs="+", help="S3 paths to PDB files")
    input_group.add_argument("--local-pdbs", nargs="+", help="Local PDB file paths to upload first")

    parser.add_argument("--input-bucket",  help="S3 bucket for uploading local PDBs")
    parser.add_argument("--output-bucket", required=True, help="S3 bucket for outputs")
    parser.add_argument("--region",        default="us-east-1")

    # Inference parameters
    parser.add_argument("--benchmark",   default="active_site_unindexed_atomic_partial_ligand")
    parser.add_argument("--num-designs", type=int, default=1)
    parser.add_argument("--config-name", default="open_source_demo")

    # Polling
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for results")
    parser.add_argument("--timeout", type=int, default=240, help="Polling timeout in minutes")

    args = parser.parse_args()

    # Upload local PDBs if needed
    if args.local_pdbs:
        if not args.input_bucket:
            raise ValueError("--input-bucket required when using --local-pdbs")
        pdb_s3_paths = [upload_local_pdb(p, args.input_bucket) for p in args.local_pdbs]
    else:
        pdb_s3_paths = args.pdbs

    parameters = {
        "benchmark":   args.benchmark,
        "num_designs": args.num_designs,
        "config_name": args.config_name,
    }

    # Submit job
    output_location = invoke_async_endpoint(
        pdb_s3_paths  = pdb_s3_paths,
        output_bucket = args.output_bucket,
        parameters    = parameters,
        region        = args.region,
    )

    # Poll for results
    if not args.no_wait:
        poll_for_output(output_location, args.region, args.timeout)
