from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class CVResult:
    oof_pred: np.ndarray
    scores: List[float]
    mean_score: float


def multiclass_logloss(y_true: np.ndarray, proba: np.ndarray, labels: List[str]) -> float:
    return float(log_loss(y_true, proba, labels=labels))


def stratified_cv_predict_proba(
    fit_predict_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y: pd.Series,
    labels: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> CVResult:
    """Generic Stratified CV wrapper that returns OOF probabilities."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(X), len(labels)), dtype=float)
    scores: List[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        proba = fit_predict_fn(X_tr, y_tr, X_va)
        oof[va_idx] = proba
        score = multiclass_logloss(y_va.values, proba, labels=labels)
        scores.append(score)
        print(f"Fold {fold}: logloss={score:.5f}")

    return CVResult(oof_pred=oof, scores=scores, mean_score=float(np.mean(scores)))
