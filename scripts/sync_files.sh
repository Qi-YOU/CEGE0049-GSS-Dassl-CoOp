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
    ["datasets/global_street_scapes.py"]="CoOp/datasets/global_street_scapes.py"
    ["datasets/glare.yaml"]="CoOp/configs/datasets/glare.yaml"
    ["datasets/lighting_condition.yaml"]="CoOp/configs/datasets/lighting_condition.yaml"
    ["datasets/weather.yaml"]="CoOp/configs/datasets/weather.yaml"
)

# Copy files with prompt if target exists
for src in "${!files_to_copy[@]}"; do
    tgt="${files_to_copy[$src]}"
    
    echo "Copying $src -> $tgt"

    # Create target directory if needed
    tgt_dir=$(dirname "$tgt")
    mkdir -p "$tgt_dir"

    if [[ -f "$tgt" ]]; then
        echo "File $tgt already exists."
        read -n 1 -s -r -p "Press any key to overwrite, or Ctrl+C to cancel."
        echo
    fi

    cp "$src" "$tgt"
    echo
done

echo "All files copied successfully."
