"""Readers shared by every stage after ETL. Pandas only, so serving stays light."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REVIEW_COLUMNS = ["review_id", "app_id", "app_name", "text", "label", "helpful"]


def load_split(processed: str | Path, split: str, limit: int | None = None, seed: int = 7) -> pd.DataFrame:
    """Read one split of the ETL output (partitioned by split=train/val/test)."""
    df = pd.read_parquet(Path(processed) / "reviews", filters=[("split", "=", split)], columns=REVIEW_COLUMNS)
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=seed)
    return df.reset_index(drop=True)


def load_all(processed: str | Path, limit: int | None = None, seed: int = 7) -> pd.DataFrame:
    df = pd.read_parquet(Path(processed) / "reviews", columns=REVIEW_COLUMNS + ["split"])
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=seed)
    return df.reset_index(drop=True)
