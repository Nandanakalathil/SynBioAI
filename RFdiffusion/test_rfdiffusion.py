import boto3
import json
import os
import sys
import argparse
import time
import uuid

parser = argparse.ArgumentParser(description='RFdiffusion Async SageMaker Client')
parser.add_argument('--pdb',         default=None,           help='Path to local PDB file')
parser.add_argument('--contigs',     required=True,          help='Contig string')
parser.add_argument('--num_designs', type=int, default=1,    help='Number of designs')
parser.add_argument('--output_dir',  default='./rfdiffusion_outputs', help='Local output directory')
parser.add_argument('--hotspot_res', default=None,           help='Hotspot residues e.g. A30,A33,A34')
parser.add_argument('--endpoint',    default='rfdiffusion-async-endpoint', help='SageMaker endpoint name')
parser.add_argument('--region',      default='us-east-1',    help='AWS region')
parser.add_argument('--bucket',      default='rfdiffusion-models-119492371915', help='S3 bucket')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

s3 = boto3.client('s3', region_name=args.region)
sm = boto3.client('sagemaker', region_name=args.region)

############################################
# FUNCTION TO SCALE ENDPOINT TO ZERO
############################################
def scale_endpoint_to_zero():
    try:
        print("\nScaling endpoint to 0 instances...")

        sm.update_endpoint_weights_and_capacities(
            EndpointName=args.endpoint,
            DesiredWeightsAndCapacities=[
                {
                    "VariantName": "default",
                    "DesiredInstanceCount": 0
                }
            ]
        )

        print("Endpoint successfully scaled to 0.")

    except Exception as e:
        print(f"Failed to scale endpoint: {e}")

############################################
# Step 1: Upload PDB to S3 if provided
############################################

s3_key = None
if args.pdb:
    if not os.path.exists(args.pdb):
        print(f"ERROR: PDB file not found: {args.pdb}")
        sys.exit(1)

    print(f"Uploading {args.pdb} to S3...")
    pdb_filename = os.path.basename(args.pdb)
    s3_key = f'pdbs/{pdb_filename}'

    s3.upload_file(args.pdb, args.bucket, s3_key)

    print(f"Uploaded to s3://{args.bucket}/{s3_key}")

else:
    print("No PDB provided - using default (unconditional generation)")

############################################
# Step 2: Upload payload to S3
############################################

run_id = str(uuid.uuid4())[:8]

payload = {
    "contigs": args.contigs,
    "num_designs": args.num_designs,
    "output_prefix": f"/tmp/rfdiffusion/{run_id}/design",
}

if s3_key:
    payload["input_pdb_s3_bucket"] = args.bucket
    payload["input_pdb_s3_key"] = s3_key

if args.hotspot_res:
    payload["hotspot_res"] = args.hotspot_res

print(f"Payload: {json.dumps(payload, indent=2)}")

input_key = f'async-inputs/input_{int(time.time())}.json'

s3.put_object(
    Bucket=args.bucket,
    Key=input_key,
    Body=json.dumps(payload),
    ContentType='application/json'
)

input_s3_uri = f's3://{args.bucket}/{input_key}'

print(f"Input saved to {input_s3_uri}")

############################################
# Step 3: Invoke async endpoint
############################################

print("Invoking async endpoint...")

runtime = boto3.client('sagemaker-runtime', region_name=args.region)

response = runtime.invoke_endpoint_async(
    EndpointName=args.endpoint,
    InputLocation=input_s3_uri,
    ContentType='application/json'
)

output_location = response['OutputLocation']

print(f"Job submitted! Output will be at: {output_location}")

############################################
# Step 4: Poll for result
############################################

print("Waiting for result", end='', flush=True)

output_bucket = output_location.split('/')[2]
output_key = '/'.join(output_location.split('/')[3:])

max_wait = 86400
interval = 30
elapsed = 0

while elapsed < max_wait:

    try:
        obj = s3.get_object(Bucket=output_bucket, Key=output_key)

        result = json.loads(obj['Body'].read().decode())

        print("\nResult received!")
        print(json.dumps(result, indent=2))

        break

    except s3.exceptions.NoSuchKey:

        print('.', end='', flush=True)

        time.sleep(interval)

        elapsed += interval

    except Exception as e:

        try:

            fail_key = output_key + '.failure'

            fail_obj = s3.get_object(Bucket=output_bucket, Key=fail_key)

            print(f"\nJob FAILED: {fail_obj['Body'].read().decode()}")

            sys.exit(1)

        except:

            print(f"\nError: {e}")

            sys.exit(1)

else:

    print("\nTimeout waiting for result")

    sys.exit(1)

############################################
# Check result status
############################################

if result.get('status') != 'success':

    print("ERROR: Inference failed")

    print(result.get('error', ''))

    sys.exit(1)

############################################
# Step 5: Download outputs
############################################

print(f"\nDownloading output PDBs to {args.output_dir}/...")

for s3_output in result.get('s3_outputs', []):

    s3_key_out = s3_output.replace(f's3://{args.bucket}/', '')

    local_output = os.path.join(
        args.output_dir,
        f"{run_id}_{os.path.basename(s3_key_out)}"
    )

    s3.download_file(args.bucket, s3_key_out, local_output)

    print(f"Downloaded: {local_output}")

print(f"\nDone! {result['num_generated']} design(s) saved to {args.output_dir}/")

print(f"Run ID: {run_id}")

############################################
# WAIT 15 MINUTES THEN SCALE ENDPOINT
############################################

print("\nWaiting 15 minutes before scaling endpoint to 0...")

time.sleep(900)

scale_endpoint_to_zero()