from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from category_encoders.target_encoder import TargetEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


@dataclass(frozen=True)
class Labels:
    ordered: List[str]

    def to_index(self, y: pd.Series) -> np.ndarray:
        mapping = {k: i for i, k in enumerate(self.ordered)}
        return y.map(mapping).values

    def to_names(self, proba: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(proba, columns=self.ordered)


def make_tfidf_linear_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    text_cols: List[str],
    max_features: int = 60000,
) -> Pipeline:
    """TF‑IDF (text) + OHE (cats) + scaled numeric -> multinomial logistic regression."""
    # Combine multiple text cols into one via a small transformer
    def _combine_text(X: pd.DataFrame) -> np.ndarray:
        # safe string concat (sklearn expects 1d array of strings)
        return (
            X[text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .values
        )

    text_pipe = Pipeline(
        steps=[
            ("combine", FunctionTransformer(_combine_text, validate=False)),
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                max_features=max_features,
                strip_accents="unicode",
            )),
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
        max_iter=500,
        solver="saga",
        n_jobs=-1,
        multi_class="multinomial",
        C=2.0,
    )

    return Pipeline([("pre", pre), ("clf", clf)])


# sklearn's FunctionTransformer is used inside make_tfidf_linear_pipeline
from sklearn.preprocessing import FunctionTransformer


def fit_predict_proba_sklearn(pipe: Pipeline, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame) -> np.ndarray:
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_va)


def make_lgbm_target_enc_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Optional[Dict] = None,
) -> Tuple[TargetEncoder, LGBMClassifier]:
    """Two-stage: TargetEncoder for high-card cats + LGBM on dense features."""
    enc = TargetEncoder(cols=categorical_cols, smoothing=0.3)
    clf = LGBMClassifier(
        objective="multiclass",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        **(params or {}),
    )
    return enc, clf


def fit_predict_proba_lgbm_target_enc(
    enc: TargetEncoder,
    clf: LGBMClassifier,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
) -> np.ndarray:
    X_tr2 = X_tr.copy()
    X_va2 = X_va.copy()
    # Fill missing
    X_tr2 = X_tr2.replace([np.inf, -np.inf], np.nan)
    X_va2 = X_va2.replace([np.inf, -np.inf], np.nan)
    X_tr2 = X_tr2.fillna(0)
    X_va2 = X_va2.fillna(0)

    enc.fit(X_tr2, y_tr)
    X_tr_enc = enc.transform(X_tr2)
    X_va_enc = enc.transform(X_va2)

    clf.fit(X_tr_enc, y_tr)
    return clf.predict_proba(X_va_enc)


def make_catboost_model(cat_cols: List[str], params: Optional[Dict] = None) -> CatBoostClassifier:
    base = dict(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        depth=8,
        learning_rate=0.08,
        iterations=3000,
        random_seed=42,
        od_type="Iter",
        od_wait=80,
        verbose=200,
    )
    if params:
        base.update(params)
    return CatBoostClassifier(**base)


def fit_predict_proba_catboost(
    model: CatBoostClassifier,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    cat_cols: List[str],
) -> np.ndarray:
    # CatBoost needs categorical feature indices
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    model.fit(X_tr, y_tr, cat_features=cat_idx)
    return model.predict_proba(X_va)
