#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
import logging
import os
import time
import io
from pathlib import Path

import boto3
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("invoke_async")

POLL_INTERVAL_SECONDS = 15
MAX_WAIT_SECONDS = 1800  # 30 min per batch

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
        f"Result not available at {output_uri} after {timeout}s."
    )

def _validate_and_cleanup_df(df: pd.DataFrame, filepath: str) -> tuple[pd.DataFrame, str | None]:
    """
    Perform row-level validation. 
    Skips rows with missing data or non-canonical amino acids.
    Returns (valid_df, error_file_path).
    """
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

    # 1. Check for missing/empty SMILES or sequence
    valid_smiles = df["SMILES"].notna() & (df["SMILES"].astype(str).str.strip() != "")
    valid_seq = df["sequence"].notna() & (df["sequence"].astype(str).str.strip() != "")
    
    # 2. Check for non-canonical amino acids (like 'X') which cause server errors
    # ESM2/CatPred usually expects standard 20 AAs. 
    canonical_aas = set("ACDEFGHIKLMNPQRSTVWY")
    def check_sequence(seq):
        if not isinstance(seq, str): return "Not a string"
        chars = set(seq.upper())
        invalid = chars - canonical_aas
        if invalid:
            return f"Contains non-canonical AAs: {', '.join(sorted(invalid))}"
        return None

    seq_errors = df["sequence"].apply(check_sequence)
    
    # Combine masks
    valid_mask = valid_smiles & valid_seq & seq_errors.isna()

    error_file = None
    if not valid_mask.all():
        invalid_df = df[~valid_mask].copy()
        
        def get_reason(row):
            reasons = []
            if pd.isna(row["SMILES"]) or str(row["SMILES"]).strip() == "":
                reasons.append("Missing SMILES")
            if pd.isna(row["sequence"]) or str(row["sequence"]).strip() == "":
                reasons.append("Missing sequence")
            else:
                err = check_sequence(row["sequence"])
                if err: reasons.append(err)
            return "; ".join(reasons)

        invalid_df["error_reason"] = invalid_df.apply(get_reason, axis=1)
        
        stem = Path(filepath).stem
        error_file = f"{stem}_errors.csv"
        invalid_df.to_csv(error_file, index=False)
        logger.warning("Skipped %d rows with issues. Details: %s", len(invalid_df), error_file)
        
        df = df[valid_mask].copy().reset_index(drop=True)
        if df.empty:
            raise ValueError(f"All rows in '{filepath}' contain errors. Check {error_file}")

    # Handle pdbpath
    if "pdbpath" not in [c.lower() for c in df.columns]:
        df["pdbpath"] = [f"seq_{i}.pdb" for i in range(len(df))]
    elif "pdbpath" not in df.columns:
        actual = [c for c in df.columns if c.lower() == "pdbpath"][0]
        df["pdbpath"] = df[actual]
    
    return df, error_file

def invoke(args: argparse.Namespace) -> None:
    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    s3 = boto3.client("s3", region_name=region)
    sm_runtime = boto3.client("sagemaker-runtime", region_name=region)

    # Handle parameters
    if getattr(args, "km", False): args.parameter = "km"
    elif getattr(args, "ki", False): args.parameter = "ki"
    elif getattr(args, "kcat", False): args.parameter = "kcat"

    if args.checkpoint_dir == "/opt/ml/model/kcat" and args.parameter != "kcat":
        args.checkpoint_dir = f"/opt/ml/model/{args.parameter}"
        logger.info("Auto-setting checkpoint-dir to: %s", args.checkpoint_dir)

    if getattr(args, "input_file", None):
        args.input_csv = args.input_file

    # Load Data
    if args.demo:
        demo_files = list(Path("demo").glob("batch_*.csv"))
        if not demo_files: raise FileNotFoundError("No demo files found.")
        args.input_csv = str(demo_files[0])
        logger.info("Demo mode: using %s", args.input_csv)

    if not args.input_csv or not os.path.exists(args.input_csv):
        raise ValueError(f"Input CSV not found: {args.input_csv}")

    df_full = pd.read_csv(args.input_csv)
    df_full, error_file_global = _validate_and_cleanup_df(df_full, args.input_csv)
    
    total_rows = len(df_full)
    batch_size = args.batch_size or total_rows
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    logger.info("Processing %d rows in %d batches (size: %d)", total_rows, num_batches, batch_size)

    all_results = []
    
    stem = Path(args.input_csv).stem
    save_path = args.save_csv or f"{stem}_result.csv"

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_rows)
        df_batch = df_full.iloc[start_idx:end_idx].copy()
        
        logger.info("--- Batch %d/%d (Rows %d to %d) ---", i+1, num_batches, start_idx, end_idx-1)
        
        payload = {
            "parameter": args.parameter,
            "checkpoint_dir": args.checkpoint_dir,
            "use_gpu": not args.no_gpu,
            "input_rows": df_batch.to_dict(orient="records"),
            "input_filename": f"{stem}_batch_{i}.csv",
        }
        
        # Upload
        key = f"catpred-inputs/{stem}_b{i}_{int(time.time())}.json"
        input_s3_uri = _upload_payload(s3, args.input_bucket, key, payload)

        # Invoke
        response = sm_runtime.invoke_endpoint_async(
            EndpointName=args.endpoint_name,
            InputLocation=input_s3_uri,
            ContentType="application/json",
            Accept="application/json",
        )
        output_location = response.get("OutputLocation")
        logger.info("Batch %d accepted. Output: %s", i+1, output_location)

        # Poll
        try:
            result = _poll_for_result(s3, output_location, timeout=args.timeout)
            if result.get("status") == "error":
                logger.error("Batch %d failed: %s", i+1, result.get("error"))
                continue
            
            output_csv_text = result.get("output_csv_text", "")
            df_out = pd.read_csv(io.StringIO(output_csv_text))
            
            # Clean up output columns
            cols_to_drop = [c for c in ["sequence", "pdbpath"] if c in df_out.columns]
            if cols_to_drop:
                df_out = df_out.drop(columns=cols_to_drop)
            
            all_results.append(df_out)
            logger.info("Batch %d completed successfully.", i+1)
            
            # Intermediate save
            pd.concat(all_results, ignore_index=True).to_csv(save_path, index=False)
            
        except Exception as e:
            logger.error("Error processing batch %d: %s", i+1, str(e))
            if i < num_batches - 1:
                logger.info("Moving to next batch...")
            continue

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(save_path, index=False)
        logger.info("Prediction complete! %d total rows saved to: %s", len(final_df), save_path)
        print(f"\nSuccess! Results saved to: {save_path}")
        if error_file_global:
            print(f"Note: Some rows were skipped due to errors. See: {error_file_global}")
    else:
        logger.error("No batches completed successfully.")

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CatPred Batch Predictor")
    p.add_argument("input_file", nargs="?", help="Local CSV file")
    p.add_argument("--endpoint-name", default=_get_env("SM_ENDPOINT_NAME", "catpred-async-endpoint"))
    p.add_argument("--region", default=_get_env("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--input-csv", dest="input_csv")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--batch-size", type=int, default=500, help="Number of rows per batch")
    p.add_argument("--kcat", action="store_true")
    p.add_argument("--km", action="store_true")
    p.add_argument("--ki", action="store_true")
    p.add_argument("--parameter", choices=["kcat", "km", "ki"], default="kcat")
    p.add_argument("--checkpoint-dir", default="/opt/ml/model/kcat")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--input-bucket", default=_get_env("S3_BUCKET", "synbioai-storage"))
    p.add_argument("--output", "-o", "--save-csv", dest="save_csv")
    p.add_argument("--timeout", type=int, default=MAX_WAIT_SECONDS)
    return p

if __name__ == "__main__":
    invoke(_build_parser().parse_args())
