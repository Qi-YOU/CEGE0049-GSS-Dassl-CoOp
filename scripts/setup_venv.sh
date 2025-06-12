#!/bin/bash

# Ensure conda commands work in non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# ==================== Global Configs ====================
DATA_ROOT="F:/datasets"
CUDA_VERSION="12.8"
PYTORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
USE_CUSTOM_PREFIX=true  # Set to false to use conda envs in default locations
VENV_NAME="GSS-DASSL-COOP"
VENV_DIR="D:/Venv/$VENV_NAME"
PATCH_DIR="./patches"
DASSL_DIR="./Dassl.pytorch"
CLIP_DIR="./CLIP"
COOP_DIR="./CoOp"


# ================== System Information ==================
echo "===================================================="
echo "                System Information                  "
echo "===================================================="

# Hostname
echo "Hostname        : $(hostname)"

# OS
echo "Operating System: $(uname)"

# CUDA version (from nvidia-smi)
cuda_smi_version=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p')
if [[ -n "$cuda_smi_version" ]] ; then
    echo "CUDA Version (nvidia-smi): $cuda_smi_version"
else
    echo "WARNING: CUDA version could not be detected from nvidia-smi!"
    echo "This may indicate that NVIDIA drivers or CUDA toolkit are not properly installed."
    echo
    read -n1 -r -p "Press any key to continue or Ctrl+C to cancel..." key
    echo
fi

# GPU info
gpu_info=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null)
if [[ $? -eq 0 ]] ; then
    echo "GPU Info: $gpu_info"
else
    echo "WARNING: GPU Info could not be detected from nvidia-smi!"
    echo "This may indicate that NVIDIA drivers or CUDA toolkit are not properly installed."
    echo
    read -n1 -r -p "Press any key to continue or Ctrl+C to cancel..." key
    echo
fi

echo "===================================================="
echo

read -n1 -r -p "Press any key to continue or Ctrl+C to cancel..." key
echo

# =================== Conda Venv Setup ===================
echo "===================================================="
echo "                 Conda Venv Setup                   "
echo "===================================================="

# Determine create args and env path
if [ "$USE_CUSTOM_PREFIX" = true ]; then
    CONDA_CREATE_ARGS="--prefix $VENV_DIR"
    ENV_PATH="$VENV_DIR"
else
    CONDA_CREATE_ARGS="-n $VENV_NAME"
    ENV_PATH="$VENV_NAME"
fi

ENV_EXISTS=false

# Check if the conda venv exists
if [ "$USE_CUSTOM_PREFIX" = true ]; then
    # Check if path exists in conda env list
    NORMALIZED_VENV_DIR="${VENV_DIR//\//\\}"  # Convert forward slashes to backslashes if executed in windows
    ENV_EXISTS=$(conda env list | awk '{print $NF}' | grep -Fx "$NORMALIZED_VENV_DIR" > /dev/null && echo true || echo false)

else
    # Check if name exists in conda env list
    ENV_EXISTS=$(conda env list | grep -w "$VENV_NAME" > /dev/null && echo true || echo false)
fi

# Remove the conda venv iff exists
if [ "$ENV_EXISTS" = true ]; then
    echo "Environment already exists at $ENV_PATH. Removing..."

    echo "conda env remove -y $CONDA_CREATE_ARGS"
    if conda env remove -y $CONDA_CREATE_ARGS; then
        echo "Successfully removed environment $ENV_PATH."
    else
        echo "ERROR: Failed to remove existing environment $ENV_PATH. Aborting."
        exit 1
    fi
else
    echo "No existing environment found at $ENV_PATH."
    echo "Proceeding to create..."
fi

# Create new conda venv
echo "Creating conda env: $ENV_PATH"
conda create -y $CONDA_CREATE_ARGS python=3.10

# Activate conda env
echo "Activating environment..."
if [ "$USE_CUSTOM_PREFIX" = true ]; then
    conda activate "$VENV_DIR"
else
    conda activate "$VENV_NAME"
fi

# Check if environment is successfully activated
if [ -n "$CONDA_PREFIX" ]; then
    echo "Environment activated: $CONDA_DEFAULT_ENV ($CONDA_PREFIX)"
else
    echo "ERROR: Failed to activate environment $ENV_PATH. Aborting."
    exit 1
fi


# ============ Python Dependency Installation ============
echo "===================================================="
echo "          Python Dependency Installation            "
echo "===================================================="

# PyTorch and torchvision installation
echo "Installing PyTorch and torchvision (CUDA $CUDA_VERSION)..."

# Format CUDA version string for URL (e.g., 12.8 -> cu128)
CUDA_VERSION_CLEANED="cu${CUDA_VERSION//./}"

# Build the pip install command
PYTORCH_INSTALL_CMD="pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/$CUDA_VERSION_CLEANED"

# Echo and run the install command
echo "Running: $PYTORCH_INSTALL_CMD"
eval $PYTORCH_INSTALL_CMD


# Check, patch and install Dassl.pytorch
if [ ! -d "$DASSL_DIR" ]; then
    echo "ERROR: $DASSL_DIR not found. Please clone the repository first."
    exit 1
fi

echo "Patching Dassl.pytorch with hotfixes..."
echo "  (Dassl uses legacy scheduler from PyTorch 1.6,"
echo "  which is deprecated since 2.2 and removed in 2.7)"
cp "$PATCH_DIR/lr_scheduler.py" "$DASSL_DIR/dassl/optim/lr_scheduler.py" || {
    echo "ERROR: Failed to patch lr_scheduler.py. Aborting."
    exit 1
}

echo "Installing Dassl.pytorch..."
pip install -r "$DASSL_DIR/requirements.txt"
pip install -e "$DASSL_DIR"


# Check and install CLIP
if [ ! -d "$CLIP_DIR" ]; then
    echo "ERROR: $CLIP_DIR not found. Please clone the repository first."
    exit 1
fi

echo "Installing CLIP..."
pip install -r "$CLIP_DIR/requirements.txt"
pip install -e "$CLIP_DIR"


# Check and install CoOp
COOP_DIR="./CoOp"
if [ ! -d "$COOP_DIR" ]; then
    echo "ERROR: $COOP_DIR not found. Please clone the repository first."
    exit 1
fi

echo "Installing CoOp..."
pip install -r "$COOP_DIR/requirements.txt"

# Install Huggingface Transformers (for Huggingface Hub access)
echo "Installing Huggingface Transformers library..."
pip install transformers

# Check for broken packages
clear
echo "Checking installed packages..."
pip check

# Deactivate conda environment
echo "Deactivating environment..."
conda deactivate
clear
echo "Setup complete. Environment $ENV_PATH is ready."
