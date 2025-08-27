"""
Modified from https://github.com/KaiyangZhou/deep-person-reid & https://github.com/KaiyangZhou/Dassl.pytorch

Enhancement: Improved Classification Evaluator
----------------------------------------------
This enhancement extends the default classification evaluator by adding support for 
precision, recall, F1, balanced accuracy, and full per-class performance reporting.

Changes:
- Computes macro & weighted precision, recall, F1, and balanced accuracy.
- Prints metrics and saves them to CSV (classification_metrics.csv).
- Saves detailed classification report if config TEST.SAVE_REPORT is True.
- Retains original accuracy, error, and per-class metrics.
"""

import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import csv
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self, verbose=True):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc

        # Macro metrics
        macro_precision = 100.0 * precision_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true),
            zero_division=0
        )
        macro_recall = 100.0 * recall_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true),
            zero_division=0
        )
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true),
            zero_division=0
        )

        # Weighted metrics
        weighted_precision = 100.0 * precision_score(
            self._y_true,
            self._y_pred,
            average="weighted",
            labels=np.unique(self._y_true),
            zero_division=0
        )
        weighted_recall = 100.0 * recall_score(
            self._y_true,
            self._y_pred,
            average="weighted",
            labels=np.unique(self._y_true),
            zero_division=0
        )
        weighted_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="weighted",
            labels=np.unique(self._y_true),
            zero_division=0
        )

        # Balanced accuracy
        balanced_acc = 100.0 * balanced_accuracy_score(
            self._y_true,
            self._y_pred,
            adjusted=True)

        # Collect metrics in results dict
        results.update({
            "accuracy": acc,
            "error_rate": err,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc
        })

        if verbose:
            print(
                "=> result\n"
                f"* total: {self._total:,}\n"
                f"* correct: {self._correct:,}\n"
                f"* accuracy: {acc:.1f}%\n"
                f"* error: {err:.1f}%\n"
                f"* macro_precision: {macro_precision:.1f}%\n"
                f"* macro_recall: {macro_recall:.1f}%\n"
                f"* macro_f1: {macro_f1:.1f}%\n"
                f"* weighted_precision: {weighted_precision:.1f}%\n"
                f"* weighted_recall: {weighted_recall:.1f}%\n"
                f"* weighted_f1: {weighted_f1:.1f}%\n"
                f"* balanced_accuracy: {balanced_acc:.1f}%"
            )

            if self._per_class_res is not None:
                labels = list(self._per_class_res.keys())
                labels.sort()

                print("=> per-class result")
                accs = []

                for label in labels:
                    classname = self._lab2cname[label]
                    res = self._per_class_res[label]
                    correct = sum(res)
                    total = len(res)
                    acc = 100.0 * correct / total
                    accs.append(acc)
                    print(
                        f"* class: {label} ({classname})\t"
                        f"total: {total:,}\t"
                        f"correct: {correct:,}\t"
                        f"acc: {acc:.1f}%"
                    )
                mean_acc = np.mean(accs)
                print(f"* average: {mean_acc:.1f}%")

                results["perclass_accuracy"] = mean_acc

            if self.cfg.TEST.COMPUTE_CMAT:
                cmat = confusion_matrix(
                    self._y_true, self._y_pred, normalize="true"
                )
                save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
                torch.save(cmat, save_path)
                print(f"Confusion matrix is saved to {save_path}")
        
            if getattr(self.cfg.TEST, "SAVE_REPORT", False):
                # Generate and save the sklearn's classification report
                report = classification_report(
                    self._y_true,
                    self._y_pred,
                    target_names=[self._lab2cname[i] for i in sorted(set(self._y_true))],
                    zero_division=0
                )
                report_path = osp.join(self.cfg.OUTPUT_DIR, "classification_report.txt")
                with open(report_path, "w") as f:
                    f.write(report)
                print(f"Classification report is saved to {report_path}")

                # Generate and save the CSV
                csv_path = osp.join(self.cfg.OUTPUT_DIR, "classification_metrics.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(list(results.keys()))
                    writer.writerow([f"{v:.2f}" if isinstance(v, float) else v for v in results.values()])
                print(f"Metrics CSV saved to {csv_path}")


        return results
