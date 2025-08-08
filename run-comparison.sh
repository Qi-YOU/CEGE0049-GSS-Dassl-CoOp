#!/bin/bash

# Compare CLIP MHAdapter with baseline models (ZeroshotCLIP, Linear_Probe, CoOp, CLIP_Adapter).
# Runs each model on selected datasets with fixed configs, logs performance, and
# produces a summary for side-by-side evaluation.

# ======= Summary log setup =======
summary_file="results/comparison_summary.txt"
echo "======== TRAINING SUMMARY ========" > "$summary_file"
echo "Start time: $(date)" >> "$summary_file"
echo "" >> "$summary_file"

function run_experiment() {
  local trainer=$1
  local dataset=$2
  local config=$3
  local loss=${4:-""}
  local class_weight=${5:-""}
  local blend_ratio=${6:-""}
  local num_heads=${7:-""}
  local seed=${8:-42}
  local extra_args=${9:-""}   # <-- holds things like "--eval-only"

  local blend_tag=""
  if [[ -n "$blend_ratio" ]]; then
    blend_tag=${blend_ratio/./}
  else
    blend_tag="NULL"
  fi

  # --------------------------------------------
  # Set output directory path for Linux or Windows
  # IMPORTANT: Manually comment/uncomment the appropriate line below based on your OS.
  # --------------------------------------------

  # For Linux users (uncomment this line and comment the Windows one)
  local outdir="/root/autodl-tmp/results/${dataset}/${trainer}-clip-vitb16"
  # For Windows users (uncomment this line and comment the Linux one)
  # local outdir="../autodl-tmp/results/${dataset}/${trainer}-clip-vitb16"

  if [[ -n "$loss" ]]; then
    outdir+="-loss_${loss}"
  fi
  if [[ -n "$class_weight" ]]; then
    outdir+="-cw_${class_weight}"
  fi
  if [[ -n "$blend_ratio" ]]; then
    outdir+="-br_${blend_tag}"
  fi
  if [[ -n "$num_heads" ]]; then
    outdir+="-mh_${num_heads}"
  fi
  if [[ -n "$seed" ]]; then
    outdir+="-sd_${seed}"
  fi

  # --------------------------------------------
  # Execute the experiment...
  # --------------------------------------------
  
  echo "Running dataset=${dataset} with trainer=${trainer}..."
  echo "=== Config: ${config} | Loss: ${loss:-none} | Class Weighting: ${class_weight:-none} | Blend Ratio: ${blend_ratio:-none} | Num Heads: ${num_heads:-none} | Seed: ${seed} ==="
  
  # Record start time
  local start_time=$(date +%s)

  # Construct Running command
  local cmd="python CoOp/train.py \
    --trainer ${trainer} \
    --dataset-config-file CoOp/configs/datasets/${dataset}.yaml \
    --config-file configs/${config}.yaml \
    --output-dir ${outdir} \
    --seed ${seed} $extra_args"

  if [[ -n "$loss" ]]; then
    cmd+=" TRAINER.LOSS.NAME ${loss}"
  fi

  if [[ -n "$class_weight" ]]; then
    cmd+=" TRAINER.LOSS.CLASS_WEIGHTING ${class_weight}"
  fi

  if [[ -n "$blend_ratio" ]]; then
    cmd+=" MODEL.BLEND_RATIO ${blend_ratio}"
  fi

  if [[ -n "$num_heads" ]]; then
    cmd+=" MODEL.NUM_HEADS ${num_heads}"
  fi

  # Echo and execute the command
  echo ">> Running command:"
  echo "$cmd"
  eval "$cmd"

  # Record end time and calculate elapsed time
  local end_time=$(date +%s)
  local elapsed=$((end_time - start_time))

  # Precompute hours, minutes, seconds to avoid redundant calculations
  local hours=$((elapsed / 3600))
  local minutes=$(((elapsed % 3600) / 60))
  local seconds=$((elapsed % 60))

  # Print to stdout
  printf "Elapsed time for %s (trainer=%s, loss=%s, weight=%s, br=%s, heads=%s): %02d:%02d:%02d\n\n" \
  "$dataset" "$trainer" "${loss:-none}" "${class_weight:-none}" "${blend_ratio:-none}" "${num_heads:-none}" \
  "$hours" "$minutes" "$seconds"

  # Append to summary file (without Params line)
  {
  echo "[$dataset] ${trainer}"
  printf "Elapsed: %02d:%02d:%02d\n" "$hours" "$minutes" "$seconds"
  echo ""
  } >> "$summary_file"
}

# ==================================================================== #

# Pre-defined pass-in arguments
datasets=("glare" "lighting_condition" "pano_status" "platform" "quality" "reflection" "view_direction" "weather")
class_weights=("default" "inverse" "uniform")
trainers_order=("ZeroshotCLIP" "Linear_Probe" "CoOp" "CLIP_Adapter" "CLIP_MHAdapter")

# Dataset-specific optimal parameters for CLIP_MHAdapter, determined through
# comprehensive grid search analysis (via run-grid-search.sh)
declare -A mhadapter_params
mhadapter_params["glare"]="ce uniform 0.8 16"
mhadapter_params["lighting_condition"]="ce uniform 0.8 8"
mhadapter_params["pano_status"]="ce uniform 0.2 4"
mhadapter_params["platform"]="ce uniform 0.8 4"
mhadapter_params["quality"]="ce uniform 0.8 4"
mhadapter_params["reflection"]="ce uniform 0.8 8"
mhadapter_params["view_direction"]="ce uniform 0.8 4"
mhadapter_params["weather"]="ce inverse 0.8 16"

# Iterate through trainers over datasets in order
for dataset in "${datasets[@]}"; do
  for trainer in "${trainers_order[@]}"; do

    # Select config file base name based on trainer type
    if [[ "$trainer" == "CLIP_MHAdapter" ]]; then
      config="vit_b16-adamw"  # Use AdamW config for CLIP_MHAdapter
    else
      config="vit_b16-sgd"    # Use SGD config for other trainers
    fi

    # Fixed random seed for all experiments
    seed=42

    # Handle different trainers differently
    case "$trainer" in

      "ZeroshotCLIP")
        run_experiment "$trainer" "$dataset" "$config" "" "" "" "" "$seed" "--eval-only"
        ;;

      "Linear_Probe"|"CoOp")
        run_experiment "$trainer" "$dataset" "$config" "" "" "" "" "$seed"
        ;;

      "CLIP_Adapter")
        for cw in "${class_weights[@]}"; do
          run_experiment "$trainer" "$dataset" "$config" "ce" "$cw" "" "" "$seed"
        done
        ;;

      "CLIP_MHAdapter")
        params=(${mhadapter_params[$dataset]})
        loss="${params[0]}"
        class_weight="${params[1]}"
        blend_ratio="${params[2]}"
        num_heads="${params[3]}"
        run_experiment "$trainer" "$dataset" "$config" "$loss" "$class_weight" "$blend_ratio" "$num_heads" "$seed"
        ;;

      *)
        echo "Warning: Unknown trainer name '$trainer'"
        ;;

    esac  # <----- this closes the case statement
  done
done
