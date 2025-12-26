from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.automl.presets.text_presets import TabularNLPAutoML


@dataclass(frozen=True)
class LamaResult:
    valid_pred: np.ndarray
    model: object


def fit_lama_tabular(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str],
    timeout: int,
    cpu_limit: int = 4,
    params: Optional[Dict] = None,
    verbose: int = 2,
) -> LamaResult:
    """Fit LAMA TabularAutoML (tabular-only)."""
    roles = {
        "target": target_col,
        "drop": drop_cols,
    }
    task = Task("multiclass")
    automl = TabularAutoML(
        task=task,
        timeout=timeout,
        cpu_limit=cpu_limit,
        general_params=params or {},
    )
    _ = automl.fit_predict(train_df, roles=roles, verbose=verbose)
    pred = automl.predict(valid_df).data
    return LamaResult(valid_pred=pred, model=automl)


def fit_lama_tabular_nlp(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str],
    text_cols: List[str],
    timeout: int,
    cpu_limit: int = 4,
    params: Optional[Dict] = None,
    text_params: Optional[Dict] = None,
    tfidf_params: Optional[Dict] = None,
    autonlp_params: Optional[Dict] = None,
    verbose: int = 2,
) -> LamaResult:
    """Fit LAMA TabularNLPAutoML (tabular + text). 

    Note: depending on configuration, it may require HuggingFace model downloads.
    Use TF‑IDF-only configuration if you run in a restricted/offline environment.
    """
    roles = {
        "target": target_col,
        "drop": drop_cols,
        "text": text_cols,
    }
    task = Task("multiclass")
    automl = TabularNLPAutoML(
        task=task,
        timeout=timeout,
        cpu_limit=cpu_limit,
        general_params=params or {},
        text_params=text_params,
        tfidf_params=tfidf_params,
        autonlp_params=autonlp_params,
    )
    _ = automl.fit_predict(train_df, roles=roles, verbose=verbose)
    pred = automl.predict(valid_df).data
    return LamaResult(valid_pred=pred, model=automl)
