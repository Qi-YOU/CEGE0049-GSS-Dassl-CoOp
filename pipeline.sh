#!/bin/bash

# Define common options
SEED=42
SLEEP_TIME=10

# Record start time
START_TIME=$(date +%s)

# Define datasets and output base dir
DATASETS=("weather" "glare" "lighting_condition")
CLASS_WEIGHTINGS=("inverse" "uniform")
TRAINER="CLIP_Adapter"
CONFIG_BASE="CoOp/configs"
TRAINER_CONFIG="configs/vit_b32.yaml"

for DATASET in "${DATASETS[@]}"; do
  for WEIGHT in "${CLASS_WEIGHTINGS[@]}"; do
    DATASET_CONFIG="${CONFIG_BASE}/datasets/${DATASET}.yaml"
    OUTPUT_DIR="results/${DATASET}/ca_vitb32_e100_bs256_s${SEED}-${WEIGHT}"

    echo "====================================="
    echo "Training on dataset: $DATASET"
    echo "Output directory: $OUTPUT_DIR"
    echo "====================================="

    python CoOp/train.py \
      --trainer "$TRAINER" \
      --dataset-config-file "$DATASET_CONFIG" \
      --config-file "$TRAINER_CONFIG" \
      --output-dir "$OUTPUT_DIR" \
      --seed $SEED \
      TRAINER.LOSS.CLASS_WEIGHTING $WEIGHT

    sleep $SLEEP_TIME
  done
done

# Record end time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Convert to HH:MM:SS
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

# Print total elapsed time
echo "====================================="
echo "Total Elapsed Time: $(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)"
