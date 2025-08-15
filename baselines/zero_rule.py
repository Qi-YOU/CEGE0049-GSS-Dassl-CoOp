"""
ZeroR Classifier: A trivial baseline that always predicts the majority class.

This module implements the Zero Rule (ZeroR) classifier, which ignores all features
and simply predicts the most frequent class from the training data. It serves as
a minimal benchmark for classification tasks.

Note: This `trainer` should always run with `--eval-only` config.

Reference: 
- Witten, I. H., et al. "Data Mining: Practical Machine Learning Tools and Techniques" (4th ed., 2016).
- Holte, R.C. Very Simple Classification Rules Perform Well on Most Commonly Used Datasets. Machine Learning 11, 63–90 (1993). https://doi.org/10.1023/A:1022631118932
- Commonly used in WEKA and scikit-learn as a baseline (DummyClassifier).
"""

import os.path as osp
from collections import Counter

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_checkpoint

class ZeroR(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.majority_class = None  # Will be set during training

    def forward(self, x):
        # Always return the majority class for any input
        batch_size = x.shape[0]
        return torch.full((batch_size, self.n_classes), 
                         fill_value=self.majority_class,
                         device=x.device)

@TRAINER_REGISTRY.register()
class ZeroR_Trainer(TrainerX):
    """ZeroR trainer that memorizes and predicts the majority class."""

    def build_model(self):
        # Determine majority class from training data
        train_labels = [x.label for x in self.dm.dataset.train_x]
        class_counts = Counter(train_labels)
        majority_class = max(class_counts, key=class_counts.get)
        n_classes = len(self.dm.dataset.classnames)

        print(f"ZeroR initialized - Majority class: {majority_class} (count={class_counts[majority_class]}/{len(train_labels)})")

        self.model = ZeroR(n_classes)
        self.model.majority_class = majority_class
        self.model.to(self.device)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        
        # CrossEntropyLoss expects raw logits (no softmax needed)
        loss = F.cross_entropy(output, label)
        
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        """No actual model to load for ZeroR, but keeping interface consistent"""
        if not directory:
            print("Note: ZeroR has no trainable parameters to load")
            return

        # This empty implementation maintains API compatibility
        pass