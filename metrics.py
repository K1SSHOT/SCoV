"""Binary metrics; positive class is virus (1). AUPRC is average precision."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or y.shape != p.shape:
        raise ValueError("Expected nonempty, aligned 1-D labels and probabilities")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    pred = (p >= threshold).astype(np.uint8)
    two_classes = np.unique(y).size == 2
    return {"Accuracy": float(accuracy_score(y, pred)),
            "Precision": float(precision_score(y, pred, zero_division=0)),
            "Recall": float(recall_score(y, pred, zero_division=0)),
            "F1": float(f1_score(y, pred, zero_division=0)),
            "AUROC": float(roc_auc_score(y, p)) if two_classes else None,
            "AUPRC": float(average_precision_score(y, p)) if (y == 1).any() else None,
            "MCC": float(matthews_corrcoef(y, pred)) if np.unique(np.concatenate([y, pred])).size > 1 else 0.0, "N": int(y.size),
            "Threshold": float(threshold)}
