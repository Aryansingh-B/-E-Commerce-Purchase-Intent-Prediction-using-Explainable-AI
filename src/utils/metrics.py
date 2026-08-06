"""Evaluation-metric helpers.

PRD §4.5 requires the full metric suite (accuracy, precision, recall, F1,
ROC-AUC, PR-AUC, confusion matrix) with PR-AUC and buyer-class recall
foregrounded, plus a value-based decision threshold instead of the default
0.5. This module centralises both.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_full_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute the complete metric suite at a given decision threshold.

    Accuracy is reported for completeness only — PR-AUC and recall on the
    buying (positive) class are the headline numbers on this imbalanced
    problem (see PRD §4.5 watch-out).
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def find_value_based_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    value_per_conversion: float,
    cost_per_intervention: float,
) -> dict:
    """Choose an operating threshold from expected business value, not 0.5.

    For a candidate threshold, every session scored above it triggers an
    intervention that costs `cost_per_intervention`. Of those, the true
    positives (tp) represent captured conversions worth `value_per_conversion`
    each — the rest (fp) are pure cost. We sweep the PR curve's thresholds and
    pick the one maximising:

        expected_value = tp * value_per_conversion - (tp + fp) * cost_per_intervention

    This is the "spend on sessions where a nudge changes the outcome"
    trade-off the business team owns (PRD §1.1); the numbers are configurable
    in config.yaml.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    best = {"threshold": 0.5, "expected_value": -np.inf}
    n_pos = int(y_true.sum())

    for i, t in enumerate(thresholds):
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        n_interventions = tp + fp
        expected_value = tp * value_per_conversion - n_interventions * cost_per_intervention
        if expected_value > best["expected_value"]:
            best = {
                "threshold": float(t),
                "expected_value": float(expected_value),
                "true_positives": tp,
                "false_positives": fp,
                "interventions_triggered": n_interventions,
                "conversions_captured": tp,
                "total_conversions": n_pos,
                "precision_at_threshold": float(precisions[i]),
                "recall_at_threshold": float(recalls[i]),
            }
    return best
