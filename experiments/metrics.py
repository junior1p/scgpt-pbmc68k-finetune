from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, normalized_mutual_info_score


def compute_label_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }
