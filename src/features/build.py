from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .text import join_list_col, build_text_stats


@dataclass(frozen=True)
class FeatureSpec:
    numeric_cols: List[str]
    categorical_cols: List[str]
    text_cols: List[str]
    datetime_cols: List[str]


def _to_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)


def build_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw JSON columns into a modeling-friendly tabular form."""
    out = df.copy()

    # Lists -> counts
    out["n_photos"] = out["photos"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    out["n_features"] = out["features"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Text columns (raw)
    out["features_text"] = out["features"].apply(join_list_col)

    # Datetime
    out["created_dt"] = _to_datetime(out, "created")
    out["created_year"] = out["created_dt"].dt.year
    out["created_month"] = out["created_dt"].dt.month
    out["created_day"] = out["created_dt"].dt.day
    out["created_hour"] = out["created_dt"].dt.hour
    out["created_dow"] = out["created_dt"].dt.dayofweek

    # Simple numeric derived
    out["price_per_bed"] = out["price"] / (out["bedrooms"] + 1.0)
    out["price_per_bath"] = out["price"] / (out["bathrooms"] + 1.0)
    out["beds_plus_baths"] = out["bedrooms"] + out["bathrooms"]

    # Address normalization
    out["street_address_clean"] = out["street_address"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    out["display_address_clean"] = out["display_address"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    # Fill missing
    for c in ["manager_id", "building_id", "street_address_clean", "display_address_clean"]:
        out[c] = out[c].fillna("UNKNOWN").astype(str)

    return out


def build_model_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, FeatureSpec]:
    """Produce a purely-tabular dataset for models (and for LAMA TabularAutoML baseline)."""
    base = build_base_dataframe(df)

    # Add cheap text statistics to keep it "tabular"
    desc_stats = build_text_stats(base, "description", "desc_")
    feat_stats = build_text_stats(base, "features_text", "feat_")
    base = pd.concat([base, desc_stats, feat_stats], axis=1)

    numeric_cols = [
        "bathrooms",
        "bedrooms",
        "price",
        "latitude",
        "longitude",
        "n_photos",
        "n_features",
        "price_per_bed",
        "price_per_bath",
        "beds_plus_baths",
        "created_year",
        "created_month",
        "created_day",
        "created_hour",
        "created_dow",
    ] + list(desc_stats.columns) + list(feat_stats.columns)

    categorical_cols = [
        "manager_id",
        "building_id",
        "street_address_clean",
        "display_address_clean",
    ]

    # keep raw text too (for non-LAMA pipelines)
    text_cols = ["description", "features_text"]

    spec = FeatureSpec(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        text_cols=text_cols,
        datetime_cols=["created_dt"],
    )

    keep_cols = ["listing_id", "interest_level"] + numeric_cols + categorical_cols + text_cols + ["created_dt"]
    keep_cols = [c for c in keep_cols if c in base.columns]
    model_df = base[keep_cols].copy()

    # Make sure numeric are numeric
    for c in numeric_cols:
        if c in model_df.columns:
            model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

    return model_df, spec
