"""
Evaluation Metrics and Scripts for Zero-Shot Classification

This file provides skeletal code for evaluating both uni-label and multi-label zero-shot classification tasks.
It is model-agnostic and can be adapted for any model output format.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================
# Uni-label Evaluation
# =========================
def evaluate_unilabel(y_true, y_pred, labels=None):
    """
    Evaluate uni-label zero-shot classification.
    Args:
        y_true: List or array of true labels
        y_pred: List or array of predicted labels
        labels: List of possible class labels (optional)
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', labels=labels)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', labels=labels)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro', labels=labels)
    metrics['classification_report'] = classification_report(y_true, y_pred, labels=labels)
    return metrics

# =========================
# Multi-label Evaluation
# =========================
def evaluate_multilabel(y_true, y_pred, threshold=0.5):
    """
    Evaluate multi-label zero-shot classification.
    Args:
        y_true: 2D array (samples x classes) of true binary labels
        y_pred: 2D array (samples x classes) of predicted scores or binary labels
        threshold: Threshold for converting scores to binary predictions
    Returns:
        Dictionary of metrics
    """
    # If y_pred contains scores, binarize
    if y_pred.dtype != np.int32 and y_pred.dtype != np.int64:
        y_pred_bin = (y_pred >= threshold).astype(int)
    else:
        y_pred_bin = y_pred
    metrics = {}
    metrics['accuracy'] = (y_true == y_pred_bin).mean()
    metrics['precision_macro'] = precision_score(y_true, y_pred_bin, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred_bin, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred_bin, average='micro', zero_division=0)
    # Optional: ROC AUC and Average Precision if scores are available
    try:
        metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred, average='macro')
        metrics['average_precision_macro'] = average_precision_score(y_true, y_pred, average='macro')
    except Exception:
        metrics['roc_auc_macro'] = None
        metrics['average_precision_macro'] = None
    return metrics

# =========================
# Example Usage
# =========================
if __name__ == "__main__":
    # Uni-label example
    y_true_uni = [0, 1, 2, 1]
    y_pred_uni = [0, 2, 2, 1]
    print("Uni-label Evaluation:")
    print(evaluate_unilabel(y_true_uni, y_pred_uni, labels=[0,1,2]))

    # Multi-label example
    y_true_multi = np.array([[1,0,1],[0,1,0],[1,1,0]])
    y_pred_multi = np.array([[0.8,0.2,0.6],[0.1,0.9,0.3],[0.7,0.6,0.4]])
    print("Multi-label Evaluation:")
    print(evaluate_multilabel(y_true_multi, y_pred_multi, threshold=0.5))
