"""
A script to parse training log files and visualize training metrics including:
- Learning curves (loss and accuracy over epochs)
- Learning rate schedule
- Confusion matrix visualization
"""

import os
import re
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay


# Define mapping dict
weather_map = {
    "clear": 0,
    "cloudy": 1,
    "foggy": 2,
    "rainy": 3,
    "snowy": 4
}

lighting_condition_map = {
    "day": 0,
    "dusk/dawn": 1,
    "night": 2
}

glare_map = {
    "no": 0,
    "yes": 1
}


def get_labels_by_dataset(dataset_name):
    """
    Return the class label list corresponding to the given dataset name.

    Args:
        dataset_name (str): Name of the dataset. Expected values include
            "weather", "lighting_condition", "glare" (case-insensitive).

    Returns:
        list or None: A list of class label strings if the dataset is recognized,
            otherwise None.
    """
    if dataset_name.lower() == "weather":
        return weather_map
    elif dataset_name.lower() == "lighting_condition":
        return lighting_condition_map
    elif dataset_name.lower() == "glare":
        return glare_map
    else:
        return None


def find_latest_log(logs):
    """
    Select the latest log file from a list based on filename sorting.

    Args:
        logs (list of str): List of log filenames.

    Returns:
        str or None: Filename of the latest log or None if empty list.
    """
    if not logs:
        return None
    # Sort logs in descending order (latest first)
    logs = sorted(logs, reverse=True)
    return logs[0]


def parse_log_file(log_path):
    """
    Parse a log file to extract epoch-wise loss, accuracy, and learning rate.

    Args:
        log_path (str): Path to the log file.

    Returns:
        tuple:
            - epoch_data (dict): {epoch: {'loss': float, 'acc': float}}
            - lr_data (dict): {epoch: {'lr_float': float, 'lr_str': str}}
    """
    epoch_last_line = {}

    with open(log_path, "r") as f:
        for line in f:
            # Regex matches lines containing epoch, loss, accuracy, and learning rate info
            m = re.search(
                r"epoch \[(\d+)/\d+\].*loss [\d\.]+ \(([\d\.]+)\).*acc [\d\.]+ \(([\d\.]+)\).*lr ([\deE\+\-\.]+)",
                line)
            if m:
                epoch = int(m.group(1))
                loss = float(m.group(2))
                acc = float(m.group(3))
                lr_str = m.group(4)
                lr = float(lr_str)
                # Store the last matched line per epoch (assumes latest info is last line of epoch)
                epoch_last_line[epoch] = (loss, acc, lr_str, lr)

    epoch_data = {}
    lr_data = {}
    for e in sorted(epoch_last_line.keys()):
        loss, acc, lr_str, lr = epoch_last_line[e]
        epoch_data[e] = {"loss": loss, "acc": acc}
        lr_data[e] = {"lr_float": lr, "lr_str": lr_str}

    return epoch_data, lr_data


def plot_learning_curve(acc_loss_dict, dataset_name, experiment_name, results_root):
    """
    Plot and save the learning curve of the training loss and accuracy curve over epochs.

    Args:
        acc_loss_dict (dict): {epoch: {'loss': float, 'acc': float}}
        dataset_name (str): Dataset name (used for title and save path).
        experiment_name (str): Experiment name (used for save path).
        results_root (str): Root directory for saving plots.
    """
    epochs = list(acc_loss_dict.keys())
    losses = [acc_loss_dict[e]["loss"] for e in epochs]
    accs = [acc_loss_dict[e]["acc"] for e in epochs]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Loss", color="tab:red", fontsize=14)
    l1, = ax1.plot(epochs, losses, color="tab:red", label="Avg Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="tab:red", labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Accuracy (%)", color="tab:blue", fontsize=14)
    l2, = ax2.plot(epochs, accs, color="tab:blue", label="Avg Acc", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=12)

    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower right", bbox_to_anchor=(1.0, 0.075), fontsize=12)

    plt.title(f"Training Curve: {dataset_name.replace('_', ' ').title()}", fontsize=14)
    fig.tight_layout()

    save_dir = os.path.join(results_root, dataset_name, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"learning_curve-{dataset_name}-{experiment_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    plt.close()


def plot_lr_curve(lr_dict, dataset_name, experiment_name, results_root):
    """
    Plot and save the learning rate curve over epochs on a logarithmic scale.

    Args:
        lr_dict (dict): {epoch: {'lr_float': float, 'lr_str': str}}
        dataset_name (str): Dataset name.
        experiment_name (str): Experiment name.
        results_root (str): Root directory for saving plots.
    """
    epochs = list(lr_dict.keys())
    lrs = [lr_dict[e]["lr_float"] for e in epochs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, lrs, label="Learning Rate", color="tab:green", linewidth=2)
    ax.set_yscale("log")

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Learning Rate (log scale)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    ax.legend(loc="upper right", fontsize=12)
    plt.title(f"Learning Rate Curve: {dataset_name.replace('_', ' ').title()}", fontsize=14)

    fig.tight_layout()

    save_dir = os.path.join(results_root, dataset_name, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"lr_curve-{dataset_name}-{experiment_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    plt.close()


def plot_confusion_matrix(cmat_path, dataset_name, experiment_name, results_root, class_labels=None):
    """
    Load and plot a confusion matrix from a .pt file.

    Args:
        cmat_path (str): Path to the confusion matrix .pt file.
        dataset_name (str): Dataset name.
        experiment_name (str): Experiment name.
        results_root (str): Root directory for saving plots.
        class_labels (list of str, optional): Class labels for display.
    """
    if not os.path.exists(cmat_path):
        print(f"Confusion matrix not found: {cmat_path}")
        return

    try:
        cmat = torch.load(cmat_path, weights_only=False)
    except Exception as e:
        print(f"Failed to load confusion matrix: {e}")
        return

    # If the loaded object is a dict with 'matrix', extract it
    if isinstance(cmat, dict) and "matrix" in cmat:
        cmat = cmat["matrix"]

    # Convert to numpy array if tensor
    cmat = cmat.numpy() if isinstance(cmat, torch.Tensor) else np.array(cmat)

    # If no class labels provided, generate default labels based on the size of confusion matrix
    if class_labels is None:
        class_labels = [f"class{i}" for i in range(cmat.shape[0])]

    # Determine display format based on dtype (integers or floats)
    if np.issubdtype(cmat.dtype, np.integer):
        values_fmt = "d"
    else:
        values_fmt = ".2f"

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=class_labels)

    disp.plot(cmap=plt.cm.Blues, values_format=values_fmt, ax=ax,
              text_kw={"fontsize": 12}, colorbar=False)
    plt.title(f"Confusion Matrix: {dataset_name.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    save_dir = os.path.join(results_root, dataset_name, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"confusion_matrix-{dataset_name}-{experiment_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    plt.close()


def main(results_root="results"):
    """
    Main function to process all datasets and experiments under the results root directory.

    Args:
        results_root (str): Root directory containing datasets and experiments.
    """

    # Collect all (dataset_name, experiment_name) combinations
    dataset_experiment_pairs = []
    for dataset_name in os.listdir(results_root):
        dataset_path = os.path.join(results_root, dataset_name)

        # Skip if the path is not a directory (e.g., a file)
        if not os.path.isdir(dataset_path):
            continue
        for experiment_name in os.listdir(dataset_path):
            experiment_path = os.path.join(dataset_path, experiment_name)

            # Skip if the path is not a directory
            if os.path.isdir(experiment_path):
                dataset_experiment_pairs.append((dataset_name, experiment_name))
    
    # Plot with tqdm progressbar
    for dataset_name, experiment_name in tqdm(dataset_experiment_pairs, desc="Plotting..."):
        # Construct the full path for the current experiment directory
        experiment_path = os.path.join(results_root, dataset_name, experiment_name)

        # Find log files named like 'log.txt' or 'log.txt-<timestamp>'
        log_files = [f for f in os.listdir(experiment_path) if f.startswith("log.txt")]

        # Skip this experiment if no log files are found
        if not log_files:
            continue

        # Pick the latest log file by sorting the filenames in descending order
        latest_log = find_latest_log(log_files)
        log_path = os.path.join(experiment_path, latest_log)

        # Parse the log file to extract accuracy, loss, and learning rate data per epoch
        acc_loss_dict, lr_dict = parse_log_file(log_path)

        # Plot and save the learning curve (accuracy and loss) for this experiment
        plot_learning_curve(acc_loss_dict, dataset_name, experiment_name, results_root)

        # Plot and save the learning rate curve for this experiment
        plot_lr_curve(lr_dict, dataset_name, experiment_name, results_root)

        # Plot and save the confusion matrix visualization if the file exists
        cmat_path = os.path.join(experiment_path, "cmat.pt")
        if os.path.isfile(cmat_path):
            plot_confusion_matrix(
                cmat_path, dataset_name, experiment_name,
                results_root, get_labels_by_dataset(dataset_name).keys()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training results from logs and confusion matrices.")
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Root directory containing datasets and experiments (default: 'results')"
    )
    args = parser.parse_args()

    main(results_root=args.results_root)
