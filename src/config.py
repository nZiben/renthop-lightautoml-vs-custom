from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = root / "data" / "raw"
    data_processed: Path = root / "data" / "processed"
    artifacts: Path = root / "artifacts"
    models: Path = artifacts / "models"
    oof: Path = artifacts / "oof"
    submissions: Path = artifacts / "submissions"


SEED: int = 42
TARGET_COL: str = "interest_level"
ID_COL: str = "listing_id"
