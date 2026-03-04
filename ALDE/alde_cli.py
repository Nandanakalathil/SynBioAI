import boto3
import json
import time
import argparse
import sys
import subprocess
import os

sm_runtime = boto3.client('sagemaker-runtime')
sm = boto3.client('sagemaker')
s3 = boto3.client('s3')
autoscaling = boto3.client('application-autoscaling')

SCALE_DOWN_COOLDOWN = 900  # 15 minutes in seconds

# ─────────────────────────────────────────────────
# Auto-Scaling Helpers
# ─────────────────────────────────────────────────

def ensure_endpoint_ready(endpoint_name):
    """Check if the endpoint has instances. If scaled to 0, wake it up and wait."""
    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    status = ep['EndpointStatus']
    
    if status != 'InService':
        print(f"  Endpoint is '{status}'. Waiting for it to become InService...")
        while status != 'InService':
            time.sleep(15)
            ep = sm.describe_endpoint(EndpointName=endpoint_name)
            status = ep['EndpointStatus']
            print(f"  Status: {status}")
    
    # Check instance count
    variants = ep.get('ProductionVariants', [])
    instance_count = variants[0].get('CurrentInstanceCount', 0) if variants else 0
    desired_count = variants[0].get('DesiredInstanceCount', 0) if variants else 0
    
    if instance_count == 0 or desired_count == 0:
        print("\n⚡ Endpoint is scaled to 0. Waking it up...")
        
        # Scale up to 1
        try:
            sm.update_endpoint_weights_and_capacities(
                EndpointName=endpoint_name,
                DesiredWeightsAndCapacities=[{
                    'VariantName': 'AllTraffic',
                    'DesiredInstanceCount': 1
                }]
            )
        except Exception as e:
            print(f"  Scale-up request: {e}")
        
        # Wait for the instance to be ready
        print("  Waiting for instance to spin up (this takes 3-5 minutes)...")
        while True:
            time.sleep(15)
            ep = sm.describe_endpoint(EndpointName=endpoint_name)
            status = ep['EndpointStatus']
            variants = ep.get('ProductionVariants', [])
            current = variants[0].get('CurrentInstanceCount', 0) if variants else 0
            print(f"  Status: {status} | Instances: {current}")
            
            if status == 'InService' and current >= 1:
                break
        
        print("Endpoint is ready!\n")
    else:
        print(f"Endpoint is ready (Instances: {instance_count})\n")

def ensure_scale_to_zero_policy(endpoint_name):
    """Ensure the auto-scaling policy is set so endpoint scales to 0 after 15 min idle."""
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    
    try:
        # Register scalable target (min=0, max=1)
        autoscaling.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=0,
            MaxCapacity=1
        )
        
        # Set scaling policy with 15-min cooldown
        autoscaling.put_scaling_policy(
            PolicyName='ALDEScaleToZero',
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': 0.5,
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleInCooldown': SCALE_DOWN_COOLDOWN,
                'ScaleOutCooldown': 60
            }
        )
        print(f"Auto-scale policy active: will scale to 0 after 15 min of inactivity.\n")
    except Exception as e:
        # Policy might already exist, that's fine
        pass

# ─────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────

def upload_payload(payload, bucket):
    key = f'inputs/{int(time.time())}.json'
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload))
    return f's3://{bucket}/{key}'

def call_endpoint(endpoint_name, payload, bucket, project_name=None, prefix='campaigns'):
    # Step 1: Make sure endpoint is awake
    ensure_endpoint_ready(endpoint_name)
    
    # Step 2: Send the request
    print(f"--- Sending Request to {endpoint_name} ---")
    input_uri = upload_payload(payload, bucket)
    
    response = sm_runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=input_uri,
        ContentType='application/json'
    )
    
    inf_id = response['InferenceId']
    output_location = response['OutputLocation']
    print(f"Request Status: SUCCESS")
    print(f"Inference ID: {inf_id}")
    print(f"Output S3: {output_location}")
    print("\n--- Streaming CloudWatch Logs (Training Progress) ---")
    
    # Step 3: Stream logs until done
    wait_for_output(inf_id, endpoint_name)
    print("\nRequest Complete.")
    print("------------------------------------------")
    
    # Step 4: Auto-sync results to local
    if project_name:
        is_setup = payload.get("mode") in ["generate_domain", "initial_sample"]
        
        # We target the specific server-nested folders to keep the local view clean
        if is_setup:
            s3_sync_path = f"{prefix}/{project_name}/data/{project_name}"
            local_sync_dir = f"{project_name}/data"
            sync_opts = "--exclude '*.pt'" # We NEVER need .pt files for setup steps locally
        else:
            # For rounds, we sync the production results folder
            s3_sync_path = f"{prefix}/{project_name}/results/{project_name}_production"
            local_sync_dir = f"{project_name}/results"
            sync_opts = "" # We need .pt briefly for extraction

        os.makedirs(local_sync_dir, exist_ok=True)
        print(f"\nAuto-syncing NEW results for {project_name}...")
        subprocess.run(f"aws s3 sync s3://{bucket}/{s3_sync_path} {local_sync_dir} {sync_opts}", shell=True)
        print(" Results sync complete.")
        
        # Step 4b: Extra processing for specific modes
        if payload.get("mode") == "execute_round":
            round_num = payload.get("round")
            print(f"Auto-extracting proposals for Round {round_num}...")
            cmd = f"conda run -n alde python alde_helpers.py extract --name {project_name} --round {round_num}"
            subprocess.run(cmd, shell=True)
            
        elif payload.get("mode") == "initial_sample":
            print(f"Cleaning initial samples CSV...")
            cmd = f"conda run -n alde python alde_helpers.py clean-initial --name {project_name}"
            subprocess.run(cmd, shell=True)
        
        # Step 4c: Final Cleanup - Ensure NO .pt or .py files stay local
        print("Cleaning up temporary local files...")
        subprocess.run(f"find {project_name} -name '*.pt' -delete 2>/dev/null", shell=True)
        subprocess.run(f"find {project_name} -name '*.py' -delete 2>/dev/null", shell=True)
    
    # Step 5: Ensure scale-to-zero policy is active for cost saving
    ensure_scale_to_zero_policy(endpoint_name)

def wait_for_output(inf_id, endpoint_name, region='us-east-1', timeout=120):
    logs = boto3.client('logs', region_name=region)
    log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    
    start_wait = time.time()
    seen_tokens = set()
    search_start = int(time.time() * 1000) - (300 * 1000)  # 5 min ago (for cold starts)
    
    while (time.time() - start_wait) < timeout:
        try:
            # Re-fetch log streams every iteration (handles cold start stream changes)
            streams = logs.describe_log_streams(
                logGroupName=log_group,
                orderBy='LastEventTime',
                descending=True,
                limit=10
            )['logStreams']
            
            # Check ALL data-log streams (old and new instances)
            for stream in streams:
                if 'data-log' not in stream['logStreamName']:
                    continue
                
                events = logs.get_log_events(
                    logGroupName=log_group,
                    logStreamName=stream['logStreamName'],
                    startTime=search_start,
                    startFromHead=True
                )['events']
                
                for event in events:
                    msg = event['message']
                    if inf_id in msg:
                        clean_msg = msg.split(']')[-1].strip() if ']' in msg else msg
                        if clean_msg not in seen_tokens:
                            print(f"  {clean_msg}")
                            seen_tokens.add(clean_msg)
                            
                            if "Inference request succeeded" in msg or "error" in msg.lower():
                                return
        except Exception:
            pass
        
        time.sleep(5)
    
    print("  Timed out waiting for logs. The request may still be processing.")
    print("  Run 'python alde_cli.py sync --name <PROJECT>' to check for results.")

# ─────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ALDE SageMaker Workflow CLI")
    parser.add_argument("--endpoint", default="alde-workflow-endpoint", help="SageMaker endpoint name")
    parser.add_argument("--bucket", default="synbioai-storage", help="S3 bucket for I/O")
    parser.add_argument("--prefix", default="async-inference", help="S3 prefix for the project")

    subparsers = parser.add_subparsers(dest="command", help="ALDE Modes")

    # Command: generate-domain
    p_domain = subparsers.add_parser("generate-domain", help="Create the design space")
    p_domain.add_argument("--name", required=True, help="Project name")
    p_domain.add_argument("--k", type=int, default=3, help="Number of mutation sites")

    # Command: initial-sample
    p_sample = subparsers.add_parser("initial-sample", help="Pick random starting variants")
    p_sample.add_argument("--name", required=True, help="Project name")
    p_sample.add_argument("--n_samples", type=int, default=96, help="Number of random samples")
    p_sample.add_argument("--seed", type=int, default=42, help="Random seed")

    # Command: execute-round
    p_round = subparsers.add_parser("execute-round", help="Run an active learning round")
    p_round.add_argument("--name", required=True, help="Project name")
    p_round.add_argument("--round", type=int, required=True, help="Round number")
    p_round.add_argument("--data_csv", required=True, help="Input CSV file (e.g. fitness_initial.csv)")
    p_round.add_argument("--batch_size", type=int, default=96, help="Number of variants to propose")
    p_round.add_argument("--obj_col", default="Fitness", help="Column name for fitness values")

    # Command: sync
    p_sync = subparsers.add_parser("sync", help="Download project files from S3 to local")
    p_sync.add_argument("--name", required=True, help="Project name")

    # Command: upload
    p_upload = subparsers.add_parser("upload", help="Upload a local CSV to S3 for the project")
    p_upload.add_argument("--name", required=True, help="Project name")
    p_upload.add_argument("--file", required=True, help="Local CSV file path to upload")

    # Command: status
    p_status = subparsers.add_parser("status", help="Check endpoint status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Status check
    if args.command == "status":
        ep = sm.describe_endpoint(EndpointName=args.endpoint)
        variants = ep.get('ProductionVariants', [{}])
        v = variants[0] if variants else {}
        print(f"Endpoint: {args.endpoint}")
        print(f"Status: {ep['EndpointStatus']}")
        print(f"Instance Count: {v.get('CurrentInstanceCount', 'N/A')}")
        print(f"Desired Instance Count: {v.get('DesiredInstanceCount', 'N/A')}")
        sys.exit(0)

    s3_proj_path = f"{args.prefix}/{args.name}"

    # Sync (no endpoint needed)
    if args.command == "sync":
        os.makedirs(f"{args.name}/data", exist_ok=True)
        os.makedirs(f"{args.name}/results", exist_ok=True)
        print(f"\n--- Syncing {args.name} from S3 ---")
        # Sync data subfolder (Exclude heavy .pt files)
        subprocess.run(f"aws s3 sync s3://{args.bucket}/{s3_proj_path}/data/{args.name} {args.name}/data --exclude '*.pt'", shell=True)
        # Sync results subfolder (Exclude heavy .pt files)
        subprocess.run(f"aws s3 sync s3://{args.bucket}/{s3_proj_path}/results/{args.name}_production {args.name}/results --exclude '*.pt'", shell=True)
        print("Sync complete.")
        sys.exit(0)

    # Upload (no endpoint needed)
    if args.command == "upload":
        filename = os.path.basename(args.file)
        s3_dest = f"s3://{args.bucket}/{s3_proj_path}/data/{args.name}/{filename}"
        print(f"\n--- Uploading {args.file} to {s3_dest} ---")
        subprocess.run(f"aws s3 cp {args.file} {s3_dest}", shell=True, check=True)
        print("Upload complete.")
        sys.exit(0)

    # Build payload and invoke endpoint
    payload = {
        "name": args.name,
        "bucket": args.bucket,
        "s3_prefix": s3_proj_path
    }

    if args.command == "generate-domain":
        payload["mode"] = "generate_domain"
        payload["k"] = args.k
    elif args.command == "initial-sample":
        payload["mode"] = "initial_sample"
        payload["n_samples"] = args.n_samples
        payload["seed"] = args.seed
    elif args.command == "execute-round":
        payload["mode"] = "execute_round"
        payload["round"] = args.round
        payload["data_csv"] = args.data_csv
        payload["batch_size"] = args.batch_size
        payload["obj_col"] = args.obj_col

    call_endpoint(args.endpoint, payload, args.bucket, project_name=args.name, prefix=args.prefix)

if __name__ == "__main__":
    main()
