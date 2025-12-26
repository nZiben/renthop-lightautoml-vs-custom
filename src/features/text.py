from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def safe_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def join_list_col(values: Iterable) -> str:
    """Join list-like values into a single string."""
    if values is None or (isinstance(values, float) and np.isnan(values)):
        return ""
    if isinstance(values, (list, tuple, set)):
        return " ".join([safe_str(v) for v in values])
    return safe_str(values)


def text_basic_stats(text: str) -> dict:
    """Cheap, model-agnostic text features."""
    t = safe_str(text)
    words = _WORD_RE.findall(t.lower())
    n_chars = len(t)
    n_words = len(words)
    n_unique = len(set(words))
    n_excl = t.count("!")
    n_caps = sum(1 for c in t if c.isupper())
    return {
        "txt_n_chars": n_chars,
        "txt_n_words": n_words,
        "txt_n_unique": n_unique,
        "txt_excl_cnt": n_excl,
        "txt_caps_ratio": (n_caps / n_chars) if n_chars else 0.0,
    }


def build_text_stats(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    stats = df[col].map(text_basic_stats).apply(pd.Series)
    stats = stats.add_prefix(prefix)
    return stats
