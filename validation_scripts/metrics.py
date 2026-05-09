"""Segmentation validation metrics for comparing ThresholdMRI against ground truth.

Computes per-class Dice coefficient, sensitivity, specificity, volume ratio,
overall accuracy, and confusion matrix.

Literature benchmarks for brain tissue segmentation Dice scores:
    Excellent: > 0.90
    Good:      0.80 - 0.90
    Acceptable: 0.70 - 0.80
    Poor:      < 0.70

Reference scores:
    ANTs Atropos (real data): WM=0.84, GM=0.79, CSF=0.64
    ANTs Atropos (BrainWeb):  WM=0.96, GM=0.95, CSF=0.94
    SPM8 (BrainWeb):          WM=0.94, GM=0.93
    FreeSurfer (mean):        ~0.90

Sources:
    - Avants et al. 2011, "An Open Source Multivariate Framework for n-Tissue Segmentation"
    - Zijdenbos et al. 1994, "Statistical Validation Based on Spatial Overlap Index"
    - Klauschen et al. 2009, "Quantitative Comparison of SPM, FSL, and Brainsuite"
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


def compute_segmentation_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    class_names: dict[int, str],
) -> dict:
    """Compute segmentation validation metrics between prediction and ground truth.

    :param prediction: Integer label array from segmentation
    :param ground_truth: Integer label array of the same shape (reference labels)
    :param class_names: Mapping from integer label values to human-readable names
    :returns: Dict with per_class metrics, overall_accuracy, confusion_matrix, class_order
    """
    if prediction.shape != ground_truth.shape:
        msg = f"Shape mismatch: prediction {prediction.shape} vs ground_truth {ground_truth.shape}"
        raise ValueError(msg)

    sorted_labels = sorted(class_names.keys())
    class_order = [class_names[k] for k in sorted_labels]
    n_classes = len(sorted_labels)
    total_voxels = prediction.size

    overall_accuracy = float(np.sum(prediction == ground_truth)) / total_voxels

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i, gt_label in enumerate(sorted_labels):
        gt_mask = ground_truth == gt_label
        for j, pred_label in enumerate(sorted_labels):
            confusion[i, j] = int(np.sum(gt_mask & (prediction == pred_label)))

    per_class: OrderedDict[str, dict[str, float]] = OrderedDict()
    for label_int in sorted_labels:
        name = class_names[label_int]
        a = prediction == label_int
        b = ground_truth == label_int

        a_count = int(np.sum(a))
        b_count = int(np.sum(b))
        intersection = int(np.sum(a & b))

        denom_dice = a_count + b_count
        d = (2.0 * intersection / denom_dice) if denom_dice > 0 else 1.0

        sensitivity = (float(intersection) / b_count) if b_count > 0 else float("nan")

        not_a_not_b = int(np.sum(~a & ~b))
        not_b_count = total_voxels - b_count
        specificity = (float(not_a_not_b) / not_b_count) if not_b_count > 0 else float("nan")

        volume_ratio = (float(a_count) / b_count) if b_count > 0 else float("nan")

        per_class[name] = {
            "dice": d,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "volume_ratio": volume_ratio,
            "pred_count": a_count,
            "true_count": b_count,
        }

    return {
        "per_class": per_class,
        "overall_accuracy": overall_accuracy,
        "confusion_matrix": confusion,
        "class_order": class_order,
    }
