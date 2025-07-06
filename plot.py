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
glare_map = {"no": 0, "yes": 1}
panoramic_status_map = {"false": 0, "true": 1}
lighting_condition_map = {"day": 0, "dusk/dawn": 1, "night": 2}
weather_map = {"clear": 0, "cloudy": 1, "foggy": 2, "rainy": 3, "snowy": 4}
platform_map = {
    "cycling surface": 0, "driving surface": 1, "fields": 2,
    "railway": 3, "tunnel": 4, "walking surface": 5
}
quality_map = {"good": 0, "slightly poor": 1, "very poor": 2}
reflection_map = {"no": 0, "yes": 1}
view_direction_map = {"front/back": 0, "side": 1}

mapping_dict = {
    "glare": glare_map,
    "panoramic_status": panoramic_status_map,
    "pano_status": panoramic_status_map,  # alias
    "lighting_condition": lighting_condition_map,
    "weather": weather_map,
    "platform": platform_map,
    "quality": quality_map,
    "reflection": reflection_map,
    "view_direction": view_direction_map,
}


def get_labels_by_dataset(dataset_name):
    """
    Return the class label list corresponding to the given dataset name.

    Args:
        dataset_name (str): Name of the dataset. Expected values include
            "weather", "lighting_condition", "glare", "quality",
            "reflection", "view_direction", "panoramic_status", "pano_status" (case-insensitive).

    Returns:
        list or None: A list of class label strings if the dataset is recognized,
            otherwise None.
    """
    return mapping_dict.get(dataset_name.lower())


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
    acc_loss_dict = {}
    lr_dict = {}

    with open(log_path, "r") as f:
        lines = f.readlines()

    epoch_to_last_batch = {}

    for line in lines:
        # Match a batch log line
        batch_match = re.match(
            r"epoch \[(\d+)/\d+\] batch \[(\d+)/(\d+)\].*?loss [\d.]+ \(([\d.]+)\) acc [\d.]+ \(([\d.]+)\) f1 [\d.]+ \(([\d.]+)\) lr ([\d.eE+-]+)",
            line
        )
        if batch_match:
            epoch = int(batch_match.group(1))
            batch_idx = int(batch_match.group(2))
            total_batches = int(batch_match.group(3))

            # Always track the last batch (highest batch_idx) for each epoch
            if epoch not in epoch_to_last_batch or batch_idx > epoch_to_last_batch[epoch]["batch_idx"]:
                epoch_to_last_batch[epoch] = {
                    "batch_idx": batch_idx,
                    "loss": float(batch_match.group(4)),
                    "acc": float(batch_match.group(5)),
                    "f1": float(batch_match.group(6)),
                    "lr": float(batch_match.group(7)),
                }
            continue

        # Match a val line
        val_match = re.match(
            r"epoch \[(\d+)/\d+\] val_acc ([\d.]+) val_err [\d.]+ val_macro_prec [\d.]+ val_macro_rec [\d.]+ val_macro_f1 ([\d.]+)",
            line
        )
        if val_match:
            epoch = int(val_match.group(1))
            val_acc = float(val_match.group(2))
            val_f1 = float(val_match.group(3))

            if epoch not in acc_loss_dict:
                acc_loss_dict[epoch] = {}

            acc_loss_dict[epoch]["val_acc"] = val_acc
            acc_loss_dict[epoch]["val_f1"] = val_f1

    # Merge training and lr info into acc_loss_dict
    for epoch, batch_info in epoch_to_last_batch.items():
        if epoch not in acc_loss_dict:
            acc_loss_dict[epoch] = {}
        acc_loss_dict[epoch]["train_loss"] = batch_info["loss"]
        acc_loss_dict[epoch]["train_acc"] = batch_info["acc"]
        acc_loss_dict[epoch]["train_f1"] = batch_info["f1"]
        lr_dict[epoch] = {
            "lr_float": batch_info["lr"],
            "lr_str": f"{batch_info['lr']:.2e}"
        }

    return acc_loss_dict, lr_dict


def plot_learning_curve(acc_loss_dict, dataset_name, experiment_name, results_root):
    """
    Plot and save the learning curve of the training loss and accuracy curve over epochs.
    Args:
        acc_loss_dict (dict): 
            {epoch: {
                'train_loss': float,
                'train_acc': float,
                'train_f1': float,
                'val_acc': float,
                'val_f1': float
            }}
        dataset_name (str): Dataset name (used for title and save path).
        experiment_name (str): Experiment name (used for save path).
        results_root (str): Root directory for saving plots.
    """
    epochs = list(acc_loss_dict.keys())
    train_losses = [acc_loss_dict[e].get("train_loss", None) for e in epochs]
    train_accs = [acc_loss_dict[e].get("train_acc", None) for e in epochs]
    train_f1s = [acc_loss_dict[e].get("train_f1", None) for e in epochs]
    val_accs = [acc_loss_dict[e].get("val_acc", None) for e in epochs]
    val_f1s = [acc_loss_dict[e].get("val_f1", None) for e in epochs]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red", fontsize=14)
    l1, = ax1.plot(epochs, train_losses, color="tab:red", label="Train Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="tab:red", labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Metric (%)", color="#008080", fontsize=14)
    l2, = ax2.plot(epochs, train_accs, color="#add8e6", label="Train Acc", linewidth=2, linestyle="-")
    l3, = ax2.plot(epochs, train_f1s, color="#90ee90", label="Train F1", linewidth=2, linestyle="--")
    l4, = ax2.plot(epochs, val_accs, color="tab:blue", label="Val Acc", linewidth=2, linestyle="-")
    l5, = ax2.plot(epochs, val_f1s, color="tab:green", label="Val F1", linewidth=2, linestyle="--")
    ax2.tick_params(axis="y", labelcolor="#008080", labelsize=12)

    # Legend: Train Loss, Train Acc & Val Acc, Train F1, Val F1
    lines = [l1, l2, l4, l3, l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower right", bbox_to_anchor=(1.0, 0.075), fontsize=12)

    plt.title(f"Learning Curve: {dataset_name.replace('_', ' ').title()}", fontsize=14)
    fig.tight_layout()

    save_dir = os.path.join(results_root, dataset_name, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"learning_curve-{dataset_name}-{experiment_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    plt.close()


def plot_learning_schedule(lr_dict, dataset_name, experiment_name, results_root):
    """
    Plot and save the learning schedule curve over epochs on a logarithmic scale.

    Args:
        lr_dict (dict): {epoch: {'lr_float': float, 'lr_str': str}}
        dataset_name (str): Dataset name.
        experiment_name (str): Experiment name.
        results_root (str): Root directory for saving plots.
    """
    epochs = list(lr_dict.keys())
    lrs = [lr_dict[e]["lr_float"] for e in epochs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, lrs, label="Learning Schedule", color="tab:orange", linewidth=2)
    ax.set_yscale("log")

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Learning Schedule (log scale)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    ax.legend(loc="upper right", fontsize=12)
    plt.title(f"Learning Schedule Curve: {dataset_name.replace('_', ' ').title()}", fontsize=14)

    fig.tight_layout()

    save_dir = os.path.join(results_root, dataset_name, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"learning_schedule-{dataset_name}-{experiment_name}.png")
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
        plot_learning_schedule(lr_dict, dataset_name, experiment_name, results_root)

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
