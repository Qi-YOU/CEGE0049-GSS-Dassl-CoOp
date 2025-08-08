"""
Modified from https://github.com/KaiyangZhou/deep-person-reid & https://github.com/KaiyangZhou/Dassl.pytorch

Enhancement: Improved Classification Evaluator
----------------------------------------------
This enhancement extends the default classification evaluator by adding support for 
precision, recall, and full per-class performance reporting.

Changes:
- Computes and prints macro-averaged precision and recall.
- Saves a detailed classification report using sklearn.metrics.classification_report.
- Report is saved to OUTPUT_DIR/classification_report.txt if TEST.SAVE_CLASS_REPORT is True.
- Retains original accuracy, error, and macro F1 metrics for consistency.
"""

import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report


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

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_precision"] = macro_precision
        results["macro_recall"] = macro_recall
        results["macro_f1"] = macro_f1

        if verbose:
            print(
                "=> result\n"
                f"* total: {self._total:,}\n"
                f"* correct: {self._correct:,}\n"
                f"* accuracy: {acc:.1f}%\n"
                f"* error: {err:.1f}%\n"
                f"* macro_precision: {macro_precision:.1f}%\n"
                f"* macro_recall: {macro_recall:.1f}%\n"
                f"* macro_f1: {macro_f1:.1f}%"
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

        return results
