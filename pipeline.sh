#!/bin/bash

# Define common options
SEED=42
SLEEP_TIME=10

# Define datasets and output base dir
DATASETS=("weather" "glare" "lighting_condition")
CLASS_WEIGHTINGS=("inverse" "uniform")
TRAINER="CLIP_Adapter"
CONFIG_BASE="CoOp/configs"
TRAINER_CONFIG="configs/vit_b32.yaml"

for DATASET in "${DATASETS[@]}"; do
  for WEIGHT in "${CLASS_WEIGHTINGS[@]}"; do
    DATASET_CONFIG="${CONFIG_BASE}/datasets/${DATASET}.yaml"
    OUTPUT_DIR="results/${DATASET}/ca_vitb32_e100_s${SEED}-${WEIGHT}"

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
