import boto3
import json
import time
import os
import argparse
import uuid
import pandas as pd
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

def invoke():
    parser = argparse.ArgumentParser(description="EvolvePro SageMaker Invocation Script")
    parser.add_argument("--fasta", type=str, required=True, help="Path to WT FASTA file")
    parser.add_argument("--labels", type=str, default=None, help="Path to lab results CSV (Round 2+)")
    parser.add_argument("--embeddings", type=str, default=None, help="Path to existing embeddings CSV (Round 2+)")
    parser.add_argument("--num_mutants", type=int, default=16, help="Number of variants to suggest")
    parser.add_argument("--max_mutations", type=int, default=1, help="Max mutations per candidate (1 for singles, 2+ for multis)")
    parser.add_argument("--search_top_n", type=int, default=20, help="How many top singles to cross-breed")
    parser.add_argument("--measured_var", type=str, default="activity", help="Column name for lab activity")
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--region", type=str, default="us-east-1")
    args = parser.parse_args()

    # 1. Initialize Clients
    custom_config = Config(
        retries={'max_attempts': 10, 'mode': 'standard'},
        connect_timeout=600,
        read_timeout=600,
        tcp_keepalive=True
    )
    s3 = boto3.client('s3', region_name=args.region, config=custom_config)
    sm_runtime = boto3.client('sagemaker-runtime', region_name=args.region)
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    bucket = f"sagemaker-{args.region}-{account_id}"
    endpoint_name = "EvolvePro-Endpoint-Async-V2"
    
    job_id = args.job_id or str(uuid.uuid4())[:8]
    fasta_base = os.path.splitext(os.path.basename(args.fasta))[0]

    # 2. Determine Mode
    if args.labels and args.embeddings:
        mode = "active-learning"
        print(f"[*] Starting ROUND X (Active Learning) for Job: {job_id}")
    else:
        mode = "zero-shot-dms"
        print(f"[*] Starting ROUND 1 (Zero-Shot Master Scan) for Job: {job_id}")

    # 3. Upload Data to S3
    config = TransferConfig(
        multipart_threshold=1024*1024*15,  # 15 Megabytes
        multipart_chunksize=1024*1024*15,
        max_concurrency=1, 
        use_threads=False
    )
    
    # Always upload FASTA
    fasta_key = f"evolvepro/inputs/{job_id}/{fasta_base}.fasta"
    s3.upload_file(args.fasta, bucket, fasta_key)
    
    payload = {
        "mode": mode,
        "fasta_s3_uri": f"s3://{bucket}/{fasta_key}",
        "num_mutants_per_round": args.num_mutants,
        "job_id": job_id,
        "measured_var": args.measured_var
    }

    if mode == "active-learning":
        label_key = f"evolvepro/inputs/{job_id}/labels.csv"
        emb_key = f"evolvepro/inputs/{job_id}/embeddings.csv"
        
        print(f"[*] Uploading labels: {args.labels}")
        df_l = pd.read_csv(args.labels)
        if args.measured_var in df_l.columns:
            if 'activity_scaled' not in df_l.columns:
                df_l['activity_scaled'] = df_l[args.measured_var]
            if 'activity_binary' not in df_l.columns:
                # Simple binary: 1 if > 0, else 0 (or just all 1s if they are all active)
                df_l['activity_binary'] = (df_l[args.measured_var] > 0).astype(int)
            df_l.to_csv(args.labels, index=False)
        s3.upload_file(args.labels, bucket, label_key)
        
        print(f"[*] Uploading embeddings ({os.path.getsize(args.embeddings)/(1024*1024):.1f} MB)...")
        s3.upload_file(args.embeddings, bucket, emb_key, Config=config)
        
        payload.update({
            "csv_s3_uri": f"s3://{bucket}/{label_key}",
            "embeddings_s3_uri": f"s3://{bucket}/{emb_key}",
            "max_mutations": args.max_mutations,
            "search_top_n": args.search_top_n
        })

    # 4. Trigger Endpoint
    request_key = f"evolvepro/requests/{job_id}_request.json"
    s3.put_object(Bucket=bucket, Key=request_key, Body=json.dumps(payload))
    
    print(f"[*] Invoking asynchronous endpoint {endpoint_name}...")
    response = sm_runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=f"s3://{bucket}/{request_key}",
        ContentType='application/json'
    )
    
    output_location = response['OutputLocation']
    print(f"[*] Job started. Polling for results at: {output_location}")

    # 5. Polling Logic
    start_time = time.time()
    while True:
        try:
            b, k = output_location.replace("s3://", "").split("/", 1)
            resp = s3.get_object(Bucket=b, Key=k)
            result = json.loads(resp['Body'].read().decode('utf-8'))
            print("\n[✔] Inference completed successfully!")
            break
        except s3.exceptions.NoSuchKey:
            elapsed = int(time.time() - start_time)
            print(f"\r    Processing on GPU... ({elapsed}s elapsed)", end="", flush=True)
            time.sleep(10)
        except Exception as e:
            # Check for error file
            try:
                err_k = k.replace('.out', '.err')
                err_resp = s3.get_object(Bucket=b, Key=err_k)
                print(f"\n[!] SageMaker Failed: {err_resp['Body'].read().decode('utf-8')}")
                return
            except: pass
            print(f"\n[!] Polling Error: {e}")
            return

    # 6. Download Results
    if mode == "zero-shot-dms":
        master_emb_uri = result.get("master_embeddings_s3_uri")
        suggested = result.get("suggested_variants", [])
        
        if master_emb_uri:
            out_emb = f"{fasta_base}_embeddings.csv"
            be, ke = master_emb_uri.replace("s3://", "").split("/", 1)
            s3.download_file(be, ke, out_emb)
            print(f"[+] Master Embeddings saved: {out_emb}")
        
        if suggested:
            out_rec = f"{fasta_base}_round1_recommendations.csv"
            pd.DataFrame({"variant": suggested}).to_csv(out_rec, index=False)
            print(f"[+] Round 1 Variants saved: {out_rec}")
            
    else:
        rec_uri = result.get("recommendations_s3_uri")
        upd_emb_uri = result.get("updated_embeddings_s3_uri")
        
        if rec_uri:
            out_rec = f"{fasta_base}_recommendations.csv"
            br, kr = rec_uri.replace("s3://", "").split("/", 1)
            s3.download_file(br, kr, out_rec)
            print(f"[+] New Recommendations saved: {out_rec}")
            
        if upd_emb_uri:
            # Sync back to local embeddings
            out_emb = args.embeddings
            bu, ku = upd_emb_uri.replace("s3://", "").split("/", 1)
            s3.download_file(bu, ku, out_emb)
            print(f"[+] Local Embeddings synced with discovered mutants: {out_emb}")

    print("\n[✔] Round Complete. Ready for Lab Validation!")

if __name__ == "__main__":
    invoke()
