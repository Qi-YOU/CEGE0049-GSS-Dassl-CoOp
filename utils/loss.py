"""
Loss function registry for modular and extensible loss implementations.

Provides a decorator `register_loss` to register loss functions by name
into the LOSS_REGISTRY dictionary for easy lookup and usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOSS_REGISTRY = {}

def register_loss(name):
    def decorator(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator


def compute_class_weights(labels, strategy, device=None):
    """
    Compute class weights based on a strategy.

    Args:
        labels: list or tensor of labels
        strategy: "inverse", "uniform", or None
        device: torch.device or None

    Returns:
        Tensor of weights or None
    """
    if labels is None:
        print("\n[WARNING] Class weighting strategy specified but no labels given; skipping weights.\n")
        return None

    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    counts = torch.tensor(counts, dtype=torch.float32)
    weights = None

    strategy = strategy.lower()
    if strategy == "inverse":
        weights = 1.0 / counts
    elif strategy == "uniform":
        class_fractions = counts / total_samples
        desired_avg_weight = 1.0 / len(unique_labels)
        weights = desired_avg_weight / class_fractions
    elif strategy in ["none", "n/a", "default", None]:
        return None
    else:
        print(f"\n[WARNING] Unsupported class weighting strategy '{strategy}'. Proceeding without weights.\n")
        return None

    weights /= weights.sum()

    if device is not None:
        weights = weights.to(device)

    return weights


@register_loss("ce")
def build_ce_loss(cfg, weights=None, device=None):
    if weights is not None:
        weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (probs * targets_onehot).sum(dim=1)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = -alpha_t * ((1 - pt) ** self.gamma) * torch.log(pt + 1e-9)
        else:
            loss = -((1 - pt) ** self.gamma) * torch.log(pt + 1e-9)
        return loss.mean()
    

@register_loss("focal")
def focal_loss(cfg, weights=None, device=None):
    gamma = getattr(cfg.TRAINER.LOSS, "FOCAL_GAMMA", 2.0)
    alpha_cfg = getattr(cfg.TRAINER.LOSS, "FOCAL_ALPHA", None)

    if alpha_cfg is not None:
        alpha = torch.tensor(alpha_cfg, dtype=torch.float32)
        if device is not None:
            alpha = alpha.to(device)
    else:
        alpha = weights  # fallback to weights if provided

    return FocalLoss(gamma=gamma, alpha=alpha)


@register_loss("cb")
def class_balanced_loss(cfg, weights=None, device=None, labels=None):
    """
    Class-Balanced Loss (Cui et al.) implementation.

    Args:
        cfg: config object with TRAINER.LOSS.CB_BETA hyperparam
        weights: optional precomputed weights (ignored here)
        device: torch.device or None
        labels: list or tensor of training labels, needed to compute class counts

    Returns:
        nn.CrossEntropyLoss with class-balanced weights
    """
    if labels is None:
        print("\n[WARNING] CB loss selected but no labels given; "
              "proceeding without class weights.\n")
        return nn.CrossEntropyLoss()

    beta = getattr(cfg.TRAINER.LOSS, "CB_BETA", 0.999)  # default to 0.999 if not set
    labels = torch.tensor(labels) if not isinstance(
        labels, torch.Tensor) else labels

    counts = torch.bincount(labels)
    counts = counts.float().clamp(min=1.0)

    effective_num = 1.0 - beta ** counts
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum()

    if device is not None:
        weights = weights.to(device)

    return nn.CrossEntropyLoss(weight=weights)


def build_loss_fn(cfg, labels=None, device=None):
    """
    Build loss function based on config.

    Args:
        cfg: config object with TRAINER.LOSS.NAME and TRAINER.LOSS.CLASS_WEIGHTING
        labels: list or tensor of training labels (used for class weights)
        device: torch.device or None

    Returns:
        loss function instance
    """
    loss_name = getattr(cfg.TRAINER.LOSS, "NAME", "ce").lower()
    class_weighting = getattr(cfg.TRAINER.LOSS, "CLASS_WEIGHTING", None)
    if class_weighting is not None:
        class_weighting = class_weighting.lower()
    else:
        class_weighting = "none"

    weights = compute_class_weights(labels, class_weighting, device=device)

    if loss_name not in LOSS_REGISTRY:
        print(f"\n[WARNING] Loss '{loss_name}' not implemented, "
              "defaulting to CrossEntropyLoss.\n")
        loss_name = "ce"

    # For class_balanced_loss, pass labels explicitly
    if loss_name == "cb":
        return LOSS_REGISTRY[loss_name](cfg, weights=weights, device=device, labels=labels)

    return LOSS_REGISTRY[loss_name](cfg, weights=weights, device=device)
