# CEGE0049-GSS-Dassl-CoOp

This repository contains the MSc Research project for CEGE0049, focused on global street scene classification using CLIP-based few-shot learning frameworks such as Dassl and CoOp. The goal is to classify scene conditions (weather, glare, and lighting) from street-level images with the dataset Global Streetscapes (Published: https://doi.org/10.1016/j.isprsjprs.2024.06.023), leveraging vision-language models.

This repository depends on:
- `Git` Required for cloning repositories and running scripts via Bash (git clone, bash setup.sh, etc.).
- `PyTorch` Core deep learning framework used for model training and inference.
- `CLIP` A vision-language model (VLM) from OpenAI used for aligning images and text in a shared embedding space.
- `Dassl.pytorch` A domain adaptation and generalization framework built on PyTorch, provides training infrastructure.
- `CoOp` A method built on top of CLIP for few-shot learning using prompt tuning.
- `Hugging Face Hub` Used to download datasets directly through its API.

## Installations

### 1. Prerequisites
- Miniconda or Anaconda
- NVIDIA GPU with CUDA 12.8 support (CUDA >= 12.8)
- Python 3.10

### 2. Clone Dependency Repository

Install these dependencies using shallow clones (--depth 1) to save time and bandwidth by skipping full git history:

```bash
# Dassl
git clone --depth 1 https://github.com/KaiyangZhou/Dassl.pytorch.git
```

```bash
# CLIP
git clone --depth 1 https://github.com/openai/CLIP.git
```

```bash
# CoOp
git clone --depth 1 https://github.com/KaiyangZhou/CoOp.git
```

### 2. Environmental Setup

- For Linux:
    Make sure the setup script is executable and then run it:
    ```bash
    # Ensure the script is executable
    chmod +x scripts/setup_venv.sh
    ```
    ```bash
    # Run the script
    ./scripts/setup_venv.sh
    ```
-  For Windows (via Git Bash or Conda Prompt):
    Open an Anaconda Prompt window, keep it in the `base` environment, and execute these commands:
    
    i. Ensure Git Bash is available in your PATH:
    ```bash
    # Add Git's Bash to PATH (adjust the path if Git is installed elsewhere)
    set PATH=C:\Program Files\Git\bin;%PATH%
    ```
    ii. Then verify Bash is available:
    ```
    bash --version
    ```
    iii. Make sure the setup script is executable and then run it:
    ```bash
    # Ensure the script is executable
    chmod +x scripts/setup_venv.sh
    ```
    ```bash
    # Run the script with Git Bash
    bash scripts/setup_venv.sh
    ```

Note: Depending on your internet speed, the setup process may take 5–10 minutes to complete.

If you encounter errors while installing packages, please double-check your network connection and the availability of relevant Conda or PyPI channels.

This script may prompt you to press a key to continue at certain steps, giving you a moment to review your system status before proceeding.

### 3. Download Dataset
TBC.