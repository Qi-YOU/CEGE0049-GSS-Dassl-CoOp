#!/bin/bash

# === Summary log setup ===
summary_file="train_summary.txt"
echo "==== TRAINING SUMMARY ====" > "$summary_file"
echo "Start time: $(date)" >> "$summary_file"
echo "" >> "$summary_file"

function run_experiment() {
  local dataset=$1
  local loss=$2
  local class_weight=$3
  local config=$4
  local blend_ratio=$5
  local num_heads=$6
  local gamma=${7:-""} # optional

  local blend_tag=${blend_ratio/./}

  # For Linux users (uncomment this line and comment the Windows one)
  local outdir="/root/autodl-tmp/results/${dataset}/clip-vitb16-mh_${num_heads}-${loss}-${class_weight}-br_${blend_tag}"

  # For Windows users (uncomment this line and comment the Linux one)
  # local outdir="../autodl-tmp/results/${dataset}/clip-vitb16-mh_${num_heads}-${loss}-${class_weight}-br_${blend_tag}"

  echo "Running ${dataset} with CLIP MHAdapter + Multihead Attention Heads #=${num_heads}..."
  echo "=== Loss: ${loss} | Weighting: ${class_weight} | Blend: ${blend_ratio} | Num_heads: ${num_heads} | Config: ${config} ==="
  local start_time=$(date +%s)

  local cmd="python CoOp/train.py \
    --trainer CLIP_MHAdapter \
    --dataset-config-file CoOp/configs/datasets/${dataset}.yaml \
    --config-file configs/${config}.yaml \
    --output-dir ${outdir} \
    --seed 42 \
    TRAINER.LOSS.NAME ${loss} \
    TRAINER.LOSS.CLASS_WEIGHTING ${class_weight} \
    MODEL.BLEND_RATIO ${blend_ratio} \
    MODEL.NUM_HEADS ${num_heads}"

  echo ">> Running command:"
  echo "$cmd"
  eval "$cmd"

  local end_time=$(date +%s)
  local elapsed=$((end_time - start_time))
  printf "Elapsed time for %s (loss=%s, weight=%s, br=%s, heads=%d): %02d:%02d:%02d\n\n" \
    "$dataset" "$loss" "$class_weight" "$blend_ratio" "$num_heads" \
    $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

  # --- Log to summary ---
  {
    echo "[$dataset] vitb16-mh-${loss}_${class_weight}"
    echo "Tag: H${num_heads}_BR${blend_ratio}"
    printf "Elapsed: %02d:%02d:%02d\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
    echo ""
  } >> "$summary_file"
}

datasets=("glare" "lighting_condition" "pano_status" "platform" "quality" "reflection" "view_direction" "weather")
loss="ce"
config="vit_b16-adamw"
class_weights=("inverse" "uniform")
blend_ratios=("0.2" "0.8")
num_heads_list=(4 8 16)

total_start=$(date +%s)

for dataset in "${datasets[@]}"; do
  for class_weight in "${class_weights[@]}"; do
    for blend_ratio in "${blend_ratios[@]}"; do
      for num_heads in "${num_heads_list[@]}"; do
        run_experiment "$dataset" "$loss" "$class_weight" "$config" "$blend_ratio" "$num_heads"
      done
    done
  done
done

total_end=$(date +%s)
total_elapsed=$((total_end - total_start))
total_hours=$((total_elapsed / 3600))
total_minutes=$(((total_elapsed % 3600) / 60))
total_seconds=$((total_elapsed % 60))

{
  echo "================================================"
  printf "Total elapsed time for all experiments: %02d:%02d:%02d\n" $total_hours $total_minutes $total_seconds
  echo "End time: $(date)"
  echo "==== END OF TRAINING ===="
} >> "$summary_file"

echo "================================================"
printf "Total elapsed time for all experiments: %02d:%02d:%02d\n" $total_hours $total_minutes $total_seconds
echo "================================================"
