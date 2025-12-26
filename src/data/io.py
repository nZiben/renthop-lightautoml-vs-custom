from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_renthop_json(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load RentHop data from Kaggle JSON files."""
    train = pd.read_json(train_path)
    test = pd.read_json(test_path)

    # Ensure consistent columns
    if "interest_level" not in test.columns:
        test["interest_level"] = pd.NA

    return train, test


def load_sample_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
