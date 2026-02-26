import boto3
import time
import os
import zipfile
import subprocess
import argparse

# ── Config ────────────────────────────────────────────────────────────────────
ENDPOINT_NAME = "omegafold-async-endpoint"
S3_BUCKET     = "filtarationmetrices"
S3_INPUT_KEY  = "input/omegafold_input.csv"
REGION        = "us-east-1"
POLL_INTERVAL = 20  # seconds

# ── Your laptop details for auto-download (Option B only) ─────────────────────
# Fill these in if you want auto-SCP to your laptop
LAPTOP_USER    = "your_username"          # e.g. "john"
LAPTOP_IP      = "your_laptop_ip"         # e.g. "192.168.1.10"
LAPTOP_SSH_KEY = "~/.ssh/id_rsa"          # SSH key EC2 uses to reach your laptop
LAPTOP_DEST    = "~/omegafold_results/"   # folder on your laptop
AUTO_SCP       = False                    # set True to enable auto-copy to laptop
# ─────────────────────────────────────────────────────────────────────────────

s3      = boto3.client("s3",                region_name=REGION)
runtime = boto3.client("sagemaker-runtime", region_name=REGION)


def submit_job(local_csv_path: str) -> str:
    print(f"\n[1/4] Uploading {local_csv_path} → s3://{S3_BUCKET}/{S3_INPUT_KEY}")
    s3.upload_file(local_csv_path, S3_BUCKET, S3_INPUT_KEY)
    print("      Upload complete.")

    print(f"\n[2/4] Invoking endpoint: {ENDPOINT_NAME}")
    response = runtime.invoke_endpoint_async(
        EndpointName  = ENDPOINT_NAME,
        InputLocation = f"s3://{S3_BUCKET}/{S3_INPUT_KEY}",
        ContentType   = "text/csv"
    )
    output_location = response["OutputLocation"]
    print(f"      Job submitted → {output_location}")
    return output_location


def wait_for_result(output_location: str) -> tuple:
    parts         = output_location.replace("s3://", "").split("/", 1)
    output_bucket = parts[0]
    output_key    = parts[1]

    print(f"\n[3/4] Waiting for OmegaFold to finish", end="", flush=True)
    while True:
        try:
            s3.head_object(Bucket=output_bucket, Key=output_key)
            print(" ✓ Done!")
            return output_bucket, output_key
        except s3.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchKey"):
                # Check for failure file
                try:
                    s3.head_object(Bucket=output_bucket, Key=output_key + ".failure")
                    print("\n[ERROR] Job failed! Downloading failure details...")
                    s3.download_file(output_bucket, output_key + ".failure", "failure.txt")
                    with open("failure.txt") as f:
                        print(f.read())
                    exit(1)
                except Exception:
                    pass
                print(".", end="", flush=True)
                time.sleep(POLL_INTERVAL)
            else:
                print(".", end="", flush=True)
                time.sleep(POLL_INTERVAL)


def download_and_extract(output_bucket: str, output_key: str, local_zip_path: str) -> str:
    print(f"\n[4/4] Downloading ZIP → {local_zip_path}")
    s3.download_file(output_bucket, output_key, local_zip_path)

    # Extract ZIP
    extract_dir = local_zip_path.replace(".zip", "_results")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(local_zip_path, "r") as zf:
        zf.extractall(extract_dir)
        extracted = zf.namelist()

    abs_dir = os.path.abspath(extract_dir)
    print(f"      Extracted {len(extracted)} files to: {abs_dir}/")
    for f in extracted:
        print(f"        - {f}")

    return abs_dir


def scp_to_laptop(local_dir: str):
    """SCP the results folder from EC2 to your laptop."""
    print(f"\n[+] Copying results to your laptop ({LAPTOP_USER}@{LAPTOP_IP})...")
    cmd = [
        "scp", "-i", LAPTOP_SSH_KEY,
        "-r", local_dir,
        f"{LAPTOP_USER}@{LAPTOP_IP}:{LAPTOP_DEST}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"    ✓ Copied to {LAPTOP_DEST} on your laptop!")
    else:
        print(f"    ✗ SCP failed: {result.stderr}")
        print(f"      You can manually copy with:")
        print(f"      scp -r ec2-user@<ec2-ip>:{local_dir} ~/omegafold_results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OmegaFold on SageMaker")
    parser.add_argument("--input",  default="test.csv",             help="Input CSV file")
    parser.add_argument("--output", default="omegafold_results.zip", help="Output ZIP path")
    args = parser.parse_args()

    # Run the pipeline
    output_location              = submit_job(args.input)
    output_bucket, output_key    = wait_for_result(output_location)
    local_dir                    = download_and_extract(output_bucket, output_key, args.output)

    print(f"\n✅ All done! Results saved to: {local_dir}/")
    print(f"   mean_plddt_results.csv — pLDDT scores")
    print(f"   *.pdb                  — 3D structure files")

    # Auto-copy to laptop if enabled
    if AUTO_SCP:
        scp_to_laptop(local_dir)
    else:
        print(f"\n💡 To download to your laptop, run this on your LOCAL machine:")
        print(f"   scp -i your-key.pem -r \\")
        print(f"     ec2-user@<your-ec2-ip>:{local_dir} \\")
        print(f"     ~/omegafold_results/")
