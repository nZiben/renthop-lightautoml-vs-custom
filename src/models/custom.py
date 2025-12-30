from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# -----------------------------
# Helpers / small components
# -----------------------------
def _combine_text_cols_factory(text_cols: List[str]):
    """Create a function for FunctionTransformer to join multiple text cols into one string."""

    def _combine_text(X: pd.DataFrame) -> np.ndarray:
        # X here is a DataFrame slice with columns=text_cols (sklearn passes DataFrame for pandas input)
        return X[text_cols].fillna("").astype(str).agg(" ".join, axis=1).values

    return _combine_text


class MulticlassTargetEncoder(BaseEstimator, TransformerMixin):
    """Fold-safe multiclass target encoding for categorical columns.

    IMPORTANT:
    - This is a transformer (not a CV loop). It becomes fold-safe automatically when used
      inside sklearn CV utilities (each fold fits on its train split).

    Output:
    - For each input categorical column, produces len(labels) numeric columns:
      P(class=label | category) with smoothing to global prior.
    """

    def __init__(self, labels: Sequence[str], smoothing: float = 20.0, missing_value: str = "UNKNOWN"):
        self.labels = list(labels)
        self.smoothing = float(smoothing)
        self.missing_value = str(missing_value)

    def fit(self, X: Any, y: Any):
        X_arr = self._to_2d_array(X)
        y_ser = pd.Series(y).astype(str)

        self.labels_ = list(self.labels)
        counts = y_ser.value_counts()
        prior = np.array([counts.get(lbl, 0) for lbl in self.labels_], dtype=float)
        prior = prior / max(prior.sum(), 1.0)
        self.prior_ = prior  # shape (K,)

        self.maps_ = []  # list of per-column dict[label -> mapping dict]
        n_cols = X_arr.shape[1]

        for j in range(n_cols):
            col = pd.Series(X_arr[:, j]).astype(str)
            col = col.replace({"nan": np.nan}).fillna(self.missing_value)

            # counts per category per class
            tab = pd.crosstab(col, y_ser).reindex(columns=self.labels_, fill_value=0)
            n = tab.sum(axis=1).values.reshape(-1, 1)  # (C,1)

            # smoothed probs: (tab + s*prior) / (n + s)
            smoothed = (tab.values + self.smoothing * self.prior_.reshape(1, -1)) / (n + self.smoothing)

            col_maps = {}
            idx = tab.index.astype(str)
            for k, lbl in enumerate(self.labels_):
                col_maps[lbl] = dict(zip(idx, smoothed[:, k].astype(float)))
            self.maps_.append(col_maps)

        self.n_in_ = n_cols
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = self._to_2d_array(X)
        if X_arr.shape[1] != getattr(self, "n_in_", X_arr.shape[1]):
            raise ValueError("Unexpected number of categorical columns in transform().")

        n_rows = X_arr.shape[0]
        k = len(self.labels_)
        out = np.zeros((n_rows, X_arr.shape[1] * k), dtype=float)

        for j in range(X_arr.shape[1]):
            col = pd.Series(X_arr[:, j]).astype(str)
            col = col.replace({"nan": np.nan}).fillna(self.missing_value)

            for t, lbl in enumerate(self.labels_):
                mapping = self.maps_[j][lbl]
                out[:, j * k + t] = col.map(mapping).fillna(self.prior_[t]).astype(float).values

        return out

    @staticmethod
    def _to_2d_array(X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X.values
        else:
            arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr


# -----------------------------
# Pipelines / estimators
# -----------------------------
def make_tfidf_logreg_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    text_cols: List[str],
    max_features: int = 60000,
) -> Pipeline:
    """TF-IDF(text) + OHE(cats) + numeric -> LogisticRegression multiclass."""
    text_pipe = Pipeline(
        steps=[
            ("combine", FunctionTransformer(_combine_text_cols_factory(text_cols), validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=max_features,
                    strip_accents="unicode",
                ),
            ),
        ]
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("text", text_pipe, text_cols),
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        solver="saga",
        max_iter=500,
        n_jobs=-1,
        C=2.0,
    )

    return Pipeline([("pre", pre), ("model", clf)])


def make_lgbm_ohe_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Tabular baseline: numeric + OHE(cats) -> LightGBM."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    base = dict(
        objective="multiclass",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    if params:
        base.update(params)

    model = LGBMClassifier(**base)
    return Pipeline([("pre", pre), ("model", model)])


def make_lgbm_targetenc_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    labels: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Tabular stronger: numeric + multiclass target encoding(cats) -> LightGBM."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # no SimpleImputer here -> keep values for our transformer
    cat_te = MulticlassTargetEncoder(labels=labels, smoothing=20.0)

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat_te", cat_te, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # output will be dense
    )

    base = dict(
        objective="multiclass",
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    if params:
        base.update(params)

    model = LGBMClassifier(**base)
    return Pipeline([("pre", pre), ("model", model)])


def make_catboost_model(params: Optional[Dict[str, Any]] = None) -> CatBoostClassifier:
    """CatBoost sklearn estimator. cat_features must be passed via fit_params in CV."""
    base = dict(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=1200,
        learning_rate=0.08,
        depth=8,
        random_seed=42,
        verbose=200,
        allow_writing_files=False,
        thread_count=-1,
    )
    if params:
        base.update(params)
    return CatBoostClassifier(**base)


def catboost_fit_params(X: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Any]:
    """Helper: build fit_params for CatBoost to mark categorical feature indices."""
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    return {"cat_features": cat_idx}
