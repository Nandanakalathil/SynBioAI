#!/bin/bash
# ALDE Cloud CLI (Dependency-Free)
# Requirements: AWS CLI configured

AWS="/usr/local/bin/aws"

ENDPOINT="alde-async-endpoint-v2"
BUCKET="synbioai-storage"
PREFIX="async-inference"

usage() {
    echo "Usage: ./alde.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  generate-domain --name NAME --k K"
    echo "  initial-sample  --name NAME --n_samples N --seed S"
    echo "  execute-round   --name NAME --round R --data_csv FILE --batch_size B"
    echo "  upload          --name NAME --file LOCAL_PATH"
    echo "  sync            --name NAME"
    echo "  clean           --name NAME"
    echo "  status"
    exit 1
}

if [ -z "$1" ]; then usage; fi

COMMAND=$1
shift

# Parse Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) NAME="$2"; shift ;;
        --k) K="$2"; shift ;;
        --n_samples) N_SAMPLES="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --round) ROUND="$2"; shift ;;
        --data_csv) DATA_CSV="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --obj_col) OBJ_COL="$2"; shift ;;
        --file) FILE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

PROJECT_S3="${PREFIX}/${NAME}"

case $COMMAND in
    status)
        "$AWS" sagemaker describe-endpoint --endpoint-name $ENDPOINT --query 'EndpointStatus'
        ;;

    upload)
        if [ -z "$NAME" ] || [ -z "$FILE" ]; then usage; fi
        FILENAME=$(basename $FILE)
        echo "Uploading $FILE to S3..."
        "$AWS" s3 cp $FILE "s3://${BUCKET}/${PROJECT_S3}/data/${NAME}/${FILENAME}"
        ;;

    sync)
        if [ -z "$NAME" ]; then usage; fi
        echo "Syncing $NAME from S3..."
        "$AWS" s3 sync "s3://${BUCKET}/${PROJECT_S3}/data/${NAME}" "${NAME}/data" --exclude "*" --include "*.csv"
        "$AWS" s3 sync "s3://${BUCKET}/${PROJECT_S3}/results/" "${NAME}/results" --exclude "*" --include "*.csv"
        ;;

    generate-domain|initial-sample|execute-round)
        if [ -z "$NAME" ]; then usage; fi

        # Build JSON Payload
        PAYLOAD="/tmp/payload.json"
        printf '{\n' > $PAYLOAD
        printf '  "name": "%s",\n' "$NAME" >> $PAYLOAD
        printf '  "bucket": "%s",\n' "$BUCKET" >> $PAYLOAD
        printf '  "s3_prefix": "%s",\n' "$PROJECT_S3" >> $PAYLOAD

        if [ "$COMMAND" == "generate-domain" ]; then
            printf '  "mode": "generate_domain",\n' >> $PAYLOAD
            printf '  "k": %s\n' "${K:-3}" >> $PAYLOAD

        elif [ "$COMMAND" == "initial-sample" ]; then
            printf '  "mode": "initial_sample",\n' >> $PAYLOAD
            printf '  "n_samples": %s,\n' "${N_SAMPLES:-96}" >> $PAYLOAD
            printf '  "seed": %s\n' "${SEED:-42}" >> $PAYLOAD

        elif [ "$COMMAND" == "execute-round" ]; then
            # Enforce round number — never silently default to 1
            if [ -z "$ROUND" ]; then
                echo "Error: --round is required for execute-round (e.g. --round 2)"
                exit 1
            fi

            # Auto-upload CSV only if --file is given
            if [ -n "$FILE" ]; then
                FILENAME=$(basename "$FILE")
                echo "Auto-uploading $FILE to S3..."
                "$AWS" s3 cp "$FILE" "s3://${BUCKET}/${PROJECT_S3}/data/${NAME}/${FILENAME}" > /dev/null
                DATA_CSV="$FILENAME"
            fi

            if [ -z "$DATA_CSV" ]; then
                echo "Error: --data_csv or --file (local path) is required for execute-round."
                exit 1
            fi

            # ── Auto-clean: drop rows with missing or non-numeric Fitness values ──
            # Determine the local CSV path to clean
            OBJ_COL_CLEAN="${OBJ_COL:-Fitness}"
            LOCAL_CSV="${NAME}/data/${DATA_CSV}"

            # If the local file exists, clean it in-place and re-upload
            if [ -f "$LOCAL_CSV" ]; then
                CLEANED=$(python3 - <<PYEOF
import csv, sys, os

infile = "$LOCAL_CSV"
tmpfile = infile + ".tmp"
obj_col = "$OBJ_COL_CLEAN"
dropped = 0

with open(infile, newline='') as fin, open(tmpfile, 'w', newline='') as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        val = row.get(obj_col, '').strip()
        try:
            float(val)
            writer.writerow(row)
        except ValueError:
            dropped += 1
            print(f"[clean] Skipping '{row.get('Combo','')}' — Fitness='{val}'", file=sys.stderr)

os.replace(tmpfile, infile)
print(dropped)
PYEOF
)
                if [ "$CLEANED" -gt 0 ] 2>/dev/null; then
                    echo "Cleaned $CLEANED non-numeric row(s) from $DATA_CSV — re-uploading..."
                    "$AWS" s3 cp "$LOCAL_CSV" "s3://${BUCKET}/${PROJECT_S3}/data/${NAME}/${DATA_CSV}" > /dev/null
                fi
            fi
            # ─────────────────────────────────────────────────────────────────────

            printf '  "mode": "execute_round",\n' >> $PAYLOAD
            printf '  "round": %s,\n' "$ROUND" >> $PAYLOAD
            printf '  "data_csv": "%s",\n' "$DATA_CSV" >> $PAYLOAD
            printf '  "batch_size": %s,\n' "${BATCH_SIZE:-96}" >> $PAYLOAD
            printf '  "obj_col": "%s"\n' "${OBJ_COL:-Fitness}" >> $PAYLOAD
        fi

        printf '}\n' >> $PAYLOAD

        # Show payload for debugging
        echo "--- Payload being sent ---"
        cat $PAYLOAD
        echo "--- End payload ---"

        # Upload Payload to S3
        INPUT_S3="s3://${BUCKET}/inputs/$(date +%s).json"
        "$AWS" s3 cp $PAYLOAD $INPUT_S3

        # Invoke Endpoint and capture JSON response
        echo "--- Invoking ALDE Cloud ($COMMAND) ---"
        RAW_RESPONSE=$("$AWS" sagemaker-runtime invoke-endpoint-async \
            --endpoint-name $ENDPOINT \
            --input-location $INPUT_S3 \
            --content-type "application/json")

        # Parse InferenceId and OutputLocation
        INF_ID=$(echo $RAW_RESPONSE | grep -o '"InferenceId": "[^"]*' | cut -d'"' -f4)
        OUTPUT_S3=$(echo $RAW_RESPONSE | grep -o '"OutputLocation": "[^"]*' | cut -d'"' -f4)

        echo "Inference ID: $INF_ID"
        echo "Output S3:    $OUTPUT_S3"
        echo "Processing started in AWS. Waiting for completion..."

        # Extract the error path (.err) from the output path (.out)
        ERROR_S3="${OUTPUT_S3%.out}.err"

        start_time=$(date +%s)
        timeout=1800 # 30 minutes

        while true; do
            # Check for success
            if "$AWS" s3 ls "$OUTPUT_S3" > /dev/null 2>&1; then
                echo ""
                echo "Success: Job completed."
                break
            fi

            # Check for error
            if "$AWS" s3 ls "$ERROR_S3" > /dev/null 2>&1; then
                echo ""
                echo "Error: Job failed in AWS."
                echo "Fetching error details..."
                "$AWS" s3 cp "$ERROR_S3" /tmp/alde_error.err > /dev/null 2>&1 && cat /tmp/alde_error.err
                exit 1
            fi

            # Check timeout
            current_time=$(date +%s)
            elapsed=$((current_time - start_time))
            if [ $elapsed -gt $timeout ]; then
                echo ""
                echo "Error: Job timed out after 30 minutes."
                exit 1
            fi

            echo -n "."
            sleep 15
        done
        echo ""

        # Auto-Sync after success
        echo "Auto-syncing results..."
        mkdir -p "${NAME}/data" "${NAME}/results"
        "$AWS" s3 sync "s3://${BUCKET}/${PROJECT_S3}/data/${NAME}" "${NAME}/data" --exclude "*" --include "*.csv"
        "$AWS" s3 sync "s3://${BUCKET}/${PROJECT_S3}/results/" "${NAME}/results" --exclude "*" --include "*.csv"
        echo "Done."
        ;;

    clean)
        if [ -z "$NAME" ]; then usage; fi
        echo "Cleaning local and S3 data for project: $NAME..."
        rm -rf "$NAME"
        "$AWS" s3 rm "s3://${BUCKET}/${PROJECT_S3}/" --recursive
        echo "Project $NAME cleaned."
        ;;
    *)
        usage
        ;;
esac
