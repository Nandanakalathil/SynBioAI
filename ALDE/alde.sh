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
        echo "{" > $PAYLOAD
        echo "  \"name\": \"$NAME\"," >> $PAYLOAD
        echo "  \"bucket\": \"$BUCKET\"," >> $PAYLOAD
        echo "  \"s3_prefix\": \"$PROJECT_S3\"," >> $PAYLOAD
        
        if [ "$COMMAND" == "generate-domain" ]; then
            echo "  \"mode\": \"generate_domain\"," >> $PAYLOAD
            echo "  \"k\": ${K:-3}" >> $PAYLOAD
        elif [ "$COMMAND" == "initial-sample" ]; then
            echo "  \"mode\": \"initial_sample\"," >> $PAYLOAD
            echo "  \"n_samples\": ${N_SAMPLES:-96}," >> $PAYLOAD
            echo "  \"seed\": ${SEED:-42}" >> $PAYLOAD
        elif [ "$COMMAND" == "execute-round" ]; then
            # Auto-upload if --file is provided
            if [ -n "$FILE" ]; then
                FILENAME=$(basename "$FILE")
                echo "Auto-uploading $FILE to S3..."
                "$AWS" s3 cp "$FILE" "s3://${BUCKET}/${PROJECT_S3}/data/${NAME}/${FILENAME}" > /dev/null
                DATA_CSV=$FILENAME
            fi
            
            if [ -z "$DATA_CSV" ]; then
                echo "Error: --data_csv or --file (local path) is required for execute-round."
                exit 1
            fi

            echo "  \"mode\": \"execute_round\"," >> $PAYLOAD
            echo "  \"round\": ${ROUND:-1}," >> $PAYLOAD
            echo "  \"data_csv\": \"$DATA_CSV\"," >> $PAYLOAD
            echo "  \"batch_size\": ${BATCH_SIZE:-96}," >> $PAYLOAD
            echo "  \"obj_col\": \"${OBJ_COL:-Fitness}\"" >> $PAYLOAD
        fi
        echo "}" >> $PAYLOAD

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
                echo "Error: Job failed in AWS. Check CloudWatch logs."
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
