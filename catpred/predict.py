#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
import logging
import os
import time
from pathlib import Path

import boto3
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("invoke_async")

POLL_INTERVAL_SECONDS = 15
MAX_WAIT_SECONDS = 1800  # 30 min



def _get_env(name: str, default: str = None) -> str:
    return os.environ.get(name, default)

def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not a valid S3 URI: {uri}")
    parts = uri[5:].split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _upload_payload(s3, bucket: str, key: str, payload: dict) -> str:
    body = json.dumps(payload, indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    uri = f"s3://{bucket}/{key}"
    logger.info("Uploaded payload to %s", uri)
    return uri


def _build_payload(args: argparse.Namespace, input_file: str | None = None) -> dict:
    """Build the JSON payload from a local CSV file."""
    filepath = input_file or args.input_csv
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    # Target columns for the core model logic
    required = {"SMILES", "sequence"}
    missing = required - set(df.columns)
    
    # Case-insensitive column mapping
    for req in missing:
        actuals = [c for c in df.columns if c.lower() == req.lower()]
        if actuals:
            df[req] = df[actuals[0]]
            missing = missing - {req}

    if missing:
        raise ValueError(f"Input CSV '{filepath}' is missing mandatory columns: {missing}")

    # Handle pdbpath: if missing, generate placeholders
    if "pdbpath" not in [c.lower() for c in df.columns]:
        logger.info("No 'pdbpath' column found. Generating placeholders...")
        df["pdbpath"] = [f"seq_{i}.pdb" for i in range(len(df))]
    elif "pdbpath" not in df.columns:
        # Standardize casing to 'pdbpath'
        actual = [c for c in df.columns if c.lower() == "pdbpath"][0]
        df["pdbpath"] = df[actual]

    return {
        "parameter": args.parameter,
        "checkpoint_dir": args.checkpoint_dir,
        "use_gpu": not args.no_gpu,
        "input_rows": df.to_dict(orient="records"),
        "input_filename": Path(filepath).name,
    }


def _poll_for_result(s3, output_uri: str, timeout: int = MAX_WAIT_SECONDS) -> dict:
    bucket, key = _parse_s3_uri(output_uri)
    logger.info("Polling for result at s3://%s/%s   (timeout: %ds)", bucket, key, timeout)
    start = time.time()
    while time.time() - start < timeout:
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            result = json.loads(obj["Body"].read().decode("utf-8"))
            logger.info("Result received after %.0fs", time.time() - start)
            return result
        except s3.exceptions.NoSuchKey:
            elapsed = time.time() - start
            logger.info("Waiting… %.0fs elapsed", elapsed)
            time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Unexpected error while polling: %s", exc)
            raise
    raise TimeoutError(
        f"Result not available at {output_uri} after {timeout}s. "
        "The job may still be running; check the S3 location manually."
    )

def invoke(args: argparse.Namespace) -> None:
    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    s3 = boto3.client("s3", region_name=region)
    sm_runtime = boto3.client("sagemaker-runtime", region_name=region)

    # Handle direct parameter flags (--km, --ki, --kcat)
    if getattr(args, "km", False):
        args.parameter = "km"
    elif getattr(args, "ki", False):
        args.parameter = "ki"
    elif getattr(args, "kcat", False):
        args.parameter = "kcat"

    # Automatically set checkpoint_dir based on parameter if not explicitly overridden
    if args.checkpoint_dir == "/opt/ml/model/kcat" and args.parameter != "kcat":
        args.checkpoint_dir = f"/opt/ml/model/{args.parameter}"
        logger.info("Auto-setting checkpoint-dir to: %s", args.checkpoint_dir)

    # If positional argument is provided, use it as input_csv
    if getattr(args, "input_file", None):
        args.input_csv = args.input_file

    if args.input_s3:
        input_s3_uri = args.input_s3
        logger.info("Using pre-existing S3 input: %s", input_s3_uri)
    elif args.input_csv:
        if not args.input_bucket:
            raise ValueError("--input-bucket or S3_BUCKET env var is required when using a local CSV")
        payload = _build_payload(args)
        stem = Path(args.input_csv).stem
        key = f"catpred-inputs/{stem}_{int(time.time())}.json"
        input_s3_uri = _upload_payload(s3, args.input_bucket, key, payload)
    elif args.demo:
        demo_files = list(Path("demo").glob("batch_*.csv"))
        if not demo_files:
            raise FileNotFoundError("No demo files found in 'demo/' directory.")
        selected = demo_files[0]
        logger.info("Demo mode: using %s", selected)
        if not args.input_bucket:
             raise ValueError("--input-bucket or S3_BUCKET env var is required for demo mode")
        payload = _build_payload(args, input_file=str(selected))
        key = f"catpred-inputs/demo_{selected.stem}_{int(time.time())}.json"
        input_s3_uri = _upload_payload(s3, args.input_bucket, key, payload)
    else:
        raise ValueError("Either --input-s3, --input-csv, or --demo must be provided.")

    # ── Determine S3 output URI ───────────────────────────────────────────────
    if args.output_s3:
        output_s3_prefix = args.output_s3.rstrip("/") + "/"
    elif args.output_bucket:
        output_s3_prefix = f"s3://{args.output_bucket}/catpred-outputs/"
    else:
        # Fallback to input bucket if only input bucket provided
        output_s3_prefix = f"s3://{args.input_bucket}/catpred-outputs/"
    input_object_name = input_s3_uri.split("/")[-1]
    expected_output_uri = f"{output_s3_prefix}{input_object_name}"

    # ── Invoke the async endpoint ─────────────────────────────────────────────
    logger.info("Invoking endpoint: %s", args.endpoint_name)
    logger.info("  Input  S3: %s", input_s3_uri)
    logger.info("  Output S3: %s", output_s3_prefix)

    response = sm_runtime.invoke_endpoint_async(
        EndpointName=args.endpoint_name,
        InputLocation=input_s3_uri,
        ContentType="application/json",
        Accept="application/json",
    )

    output_location = response.get("OutputLocation", expected_output_uri)
    logger.info("Async job accepted. Output will appear at: %s", output_location)

    # ── Poll for result ───────────────────────────────────────────────────────
    result = _poll_for_result(s3, output_location, timeout=args.timeout)

    if result.get("status") == "error":
        logger.error("Prediction failed:\n%s", result.get("error"))
        logger.error("Traceback:\n%s", result.get("traceback", ""))
        raise RuntimeError(result.get("error", "Unknown error from CatPred endpoint."))

    # ── Save / display result ─────────────────────────────────────────────────
    output_csv_text = result.get("output_csv_text", "")
    output_filename = result.get("output_filename", "predictions.csv")
    num_preds = result.get("num_predictions", "?")

    if args.save_csv:
        save_path = args.save_csv
    else:
        # Use <inputname>_result.csv as per user request
        if args.input_csv:
            stem = Path(args.input_csv).stem
            save_path = f"{stem}_result.csv"
        else:
            save_path = output_filename

    import io
    df_out = pd.read_csv(io.StringIO(output_csv_text))
    
    # Drop sequence and pdbpath if they exist
    cols_to_drop = [c for c in ["sequence", "pdbpath"] if c in df_out.columns]
    if cols_to_drop:
        df_out = df_out.drop(columns=cols_to_drop)

    df_out.to_csv(save_path, index=False)

    logger.info("Saved %s predictions to %s", num_preds, save_path)
    print(f"\nPrediction complete! {num_preds} rows saved to: {save_path}")

    # Print first few rows for a quick sanity check
    df = pd.read_csv(save_path)
    print("\nPreview (first 5 rows):")
    print(df.head(5).to_string(index=False))

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Invoke CatPred SageMaker async inference endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--endpoint-name", 
                   default=_get_env("SM_ENDPOINT_NAME", "catpred-async-endpoint"),
                   dest="endpoint_name",
                   help="Name of the SageMaker endpoint")
    p.add_argument("--region", default=_get_env("AWS_DEFAULT_REGION", "us-east-1"))

    # Positional argument for simplified usage
    p.add_argument("input_file", nargs="?", help="Local CSV to convert & upload")

    # Input: either a pre-existing S3 payload, a local CSV, or demo mode
    input_group = p.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input-s3", dest="input_s3", help="S3 URI of the JSON payload")
    input_group.add_argument("--input-csv", dest="input_csv", help="Local CSV to convert & upload")
    input_group.add_argument("--demo", action="store_true", help="Run on a sample file from demo/ directory")

    # Shortcuts for common parameters
    param_group = p.add_mutually_exclusive_group()
    param_group.add_argument("--kcat", action="store_true", help="Predict turnover frequency (default)")
    param_group.add_argument("--km", action="store_true", help="Predict Michaelis constant")
    param_group.add_argument("--ki", action="store_true", help="Predict inhibition constant")

    # Fallback/Advanced parameter control
    p.add_argument("--parameter", choices=["kcat", "km", "ki"], default="kcat", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", default="/opt/ml/model/kcat")
    p.add_argument("--no-gpu", dest="no_gpu", action="store_true")
    p.add_argument("--input-bucket", 
                   default=_get_env("S3_BUCKET", "synbioai-storage"),
                   dest="input_bucket", 
                   help="S3 bucket for uploading the payload")

    # Output
    output_group = p.add_mutually_exclusive_group(required=False)
    output_group.add_argument("--output-s3", dest="output_s3", help="S3 URI prefix for output")
    output_group.add_argument("--output-bucket", 
                               default=_get_env("S3_OUTPUT_BUCKET", _get_env("S3_BUCKET", "synbioai-storage")),
                               dest="output_bucket", 
                               help="S3 bucket for output")

    p.add_argument("--output", "-o", "--save-csv", dest="save_csv", 
                   help="Local path to save the result CSV (e.g. predictions.csv)")
    p.add_argument("--timeout", type=int, default=MAX_WAIT_SECONDS, help="Max wait seconds")
    return p


if __name__ == "__main__":
    invoke(_build_parser().parse_args())
