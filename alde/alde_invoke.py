import boto3
import pandas as pd
import time
import os
import argparse
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer

def test_alde_endpoint(csv_path, output_path="results_from_endpoint.csv", rounds=4, endpoint_name="alde-async-endpoint"):
    s3_bucket = "sagemaker-us-east-1-119492371915"
    s3_prefix = "alde-async-output/input"
    
    # Initialize session and clients
    sm_runtime = boto3.client('sagemaker-runtime')
    s3_client = boto3.client('s3')
    
    # 1. Upload input CSV to S3
    file_name = os.path.basename(csv_path)
    s3_input_key = f"{s3_prefix}/{int(time.time())}_{file_name}"
    s3_input_uri = f"s3://{s3_bucket}/{s3_input_key}"
    
    print(f"Uploading {csv_path} to {s3_input_uri}...")
    s3_client.upload_file(csv_path, s3_bucket, s3_input_key)
    
    # 2. Invoke Endpoint
    print(f"Invoking endpoint {endpoint_name}...")
    response = sm_runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=s3_input_uri,
        ContentType='text/csv',
        Accept='text/csv',
        CustomAttributes=f'X-Opt-Rounds={rounds}'
    )
    
    output_location = response['OutputLocation']
    print(f"Inference started. Output will be at: {output_location}")
    
    # 3. Poll for result
    print("Waiting for results (this may take 15-20 minutes for large datasets)...")
    output_bucket = output_location.split('/')[2]
    output_key = '/'.join(output_location.split('/')[3:])
    
    start_time = time.time()
    while True:
        try:
            s3_client.head_object(Bucket=output_bucket, Key=output_key)
            print("\nSuccess! Result found.")
            break
        except:
            elapsed = int(time.time() - start_time)
            print(f"Still waiting... ({elapsed}s)", end='\r')
            time.sleep(10)
            if elapsed > 3600: # 60 mins timeout for larger runs
                print("\nTimeout waiting for response.")
                return

    # 4. Download and show results
    s3_client.download_file(output_bucket, output_key, output_path)
    
    df_results = pd.read_csv(output_path)
    print(f"\n--- RECOMMENDED VARIANTS (Total: {len(df_results)}) ---")
    print(df_results.head(10))
    print(f"\nAll results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="results_from_endpoint.csv", help="Local path for output CSV")
    parser.add_argument("--rounds", type=int, default=4, help="Number of rounds to recommend (default: 4)")
    parser.add_argument("--endpoint", type=str, default="alde-async-endpoint-v2", help="SageMaker endpoint name")
    args = parser.parse_args()
    
    test_alde_endpoint(args.csv, output_path=args.output, rounds=args.rounds, endpoint_name=args.endpoint)
