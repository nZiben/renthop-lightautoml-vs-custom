from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

import inspect


@dataclass(frozen=True)
class CVResult:
    """OOF probabilities + fold scores computed by sklearn CV utilities."""
    oof_pred: np.ndarray
    scores: List[float]
    mean_score: float
    oof_score: float


def _cv_kwargs_for_fit_params(func: Any, fit_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    sklearn changed API:
      - older: cross_validate(..., fit_params=...)
      - newer: cross_validate(..., params=...)
    Same story for cross_val_predict.
    """
    if not fit_params:
        return {}

    sig = inspect.signature(func)
    if "params" in sig.parameters:
        return {"params": fit_params}
    if "fit_params" in sig.parameters:
        return {"fit_params": fit_params}

    # nothing supported (should be rare) -> just don't pass
    return {}


def evaluate_multiclass_cv(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    labels: List[str],
    n_splits: int = 5,
    seed: int = 42,
    n_jobs: int = -1,
    fit_params: Optional[Dict[str, Any]] = None,
) -> CVResult:
    """Multiclass CV evaluation WITHOUT manual fold loops.

    Uses:
      - cross_validate (per-fold scores)
      - cross_val_predict (OOF predict_proba)

    fit_params: optional params routed to estimator.fit on each fold
      (e.g., CatBoost needs cat_features).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # --- per-fold logloss ---
    cv_kwargs = _cv_kwargs_for_fit_params(cross_validate, fit_params)
    cv_out = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
        **cv_kwargs,
    )
    fold_scores = (-cv_out["test_score"]).astype(float).tolist()
    mean_score = float(np.mean(fold_scores))

    # --- OOF probabilities ---
    pred_kwargs = _cv_kwargs_for_fit_params(cross_val_predict, fit_params)
    oof = cross_val_predict(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        method="predict_proba",
        n_jobs=n_jobs,
        **pred_kwargs,
    )
    oof_score = float(log_loss(y, oof, labels=labels))

    return CVResult(
        oof_pred=oof,
        scores=fold_scores,
        mean_score=mean_score,
        oof_score=oof_score,
    )
