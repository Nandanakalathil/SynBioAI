#!/bin/bash
# BIRAC SynBioAI - Interactive S3 Cleanup Script

S3_BUCKET="synbioai-storage"
PREFIX="async-inference/"
REGION="us-east-1"

# List files and count
FILE_LIST=$(aws s3 ls "s3://$S3_BUCKET/$PREFIX" --recursive --human-readable --region "$REGION")

if [[ -z "$FILE_LIST" ]]; then
    echo "Bucket is already empty. No cleanup needed."
    exit 0
fi

echo "$FILE_LIST"
echo "------------------------------------------------"
COUNT=$(echo "$FILE_LIST" | wc -l | xargs)
echo "Found $COUNT files in the async-inference folder."

read -p "Are you sure you want to PERMANENTLY delete all these files? (y/N): " CONFIRM

if [[ "$CONFIRM" == "y" || "$CONFIRM" == "Y" ]]; then
    aws s3 rm "s3://$S3_BUCKET/$PREFIX" --recursive --region "$REGION"
    echo "Success: S3 temporary data cleared."
else
    echo "Cleanup cancelled. No files were deleted."
fi
