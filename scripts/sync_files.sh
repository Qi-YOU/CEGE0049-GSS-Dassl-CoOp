#!/bin/bash

# Run as: scripts/sync_files.sh

# Print current working directory
echo "Current working directory: $(pwd)"

# Check for required files/folders
if [[ -d "CLIP" && -d "CoOp" && -d "Dassl.pytorch" && -f "README.md" ]]; then
    echo "Directory check passed. Proceeding with file sync..."
    echo
else
    echo "Required directories(CLIP / CoOp / Dassl.pytorch) or README.md not found."
    echo "Make sure you are in the correct root directory of the project."
    exit 1
fi

# Define associative array of source → target file paths
declare -A files_to_copy=(
    # Dataset configurations
    ["datasets/global_street_scapes.py"]="CoOp/datasets/global_street_scapes.py"
    ["datasets/platform.yaml"]="CoOp/configs/datasets/platform.yaml"
    ["datasets/weather.yaml"]="CoOp/configs/datasets/weather.yaml"
    ["datasets/view_direction.yaml"]="CoOp/configs/datasets/view_direction.yaml"
    ["datasets/lighting_condition.yaml"]="CoOp/configs/datasets/lighting_condition.yaml"
    ["datasets/pano_status.yaml"]="CoOp/configs/datasets/pano_status.yaml"
    ["datasets/quality.yaml"]="CoOp/configs/datasets/quality.yaml"
    ["datasets/glare.yaml"]="CoOp/configs/datasets/glare.yaml"
    ["datasets/reflection.yaml"]="CoOp/configs/datasets/reflection.yaml"
    
    # Baseline models
    ["baselines/clip_adapter.py"]="CoOp/trainers/clip_adapter.py"
    ["baselines/linear_probe.py"]="CoOp/trainers/linear_probe.py"

    # The proposed model 
    ["clip_mhadapter.py"]="CoOp/trainers/clip_mhadapter.py"

    # Modified Zero-Shot CLIP with classname normalization for GlobalStreetScapes
    ["patches/trainers/zsclip.py"]="CoOp/trainers/zsclip.py"

    # Utils
    ["utils/loss.py"]="CoOp/trainers/utils/loss.py"
    ["utils/attn.py"]="CoOp/trainers/utils/attn.py"
    ["utils/__init__.py"]="CoOp/trainers/utils/__init__.py"
)

# Copy files with prompt if target exists
for src in "${!files_to_copy[@]}"; do
    tgt="${files_to_copy[$src]}"
    
    echo "Copying $src -> $tgt"

    # Create target directory if needed
    tgt_dir=$(dirname "$tgt")
    mkdir -p "$tgt_dir"

    if [[ -f "$tgt" ]]; then
        src_hash=$(sha256sum "$src" | cut -d' ' -f1)
        tgt_hash=$(sha256sum "$tgt" | cut -d' ' -f1)
        if [[ "$src_hash" == "$tgt_hash" ]]; then
            echo "  Info: $tgt is identical to $src (sha256sum match). Skipping copy."
            continue
        else
            echo "  Warning: $tgt exists and differs from $src. Overwriting."
        fi
    else
        echo "  Note: $tgt does not exist. Copying new file."
    fi

    if ! cp "$src" "$tgt"; then
        echo "  Error: Failed to copy $src to $tgt"
    fi
done

echo ""
echo "All files copied successfully."
