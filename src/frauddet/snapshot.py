"""Helpers for reading the frozen Phase 3 parquet snapshot.

Phase 3 and later feature work must read versioned snapshot parquet, not live
Mongo and not the mutable top-level Phase 2 outputs. This module keeps that path
choice in one place.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .flatten import OUTPUT_FILES


def snapshot_dir() -> Path:
    """Return the configured frozen Phase 3 snapshot directory."""
    raw = config.load_config()["phase3"]["input_dir"]
    path = Path(raw)
    if not path.is_absolute():
        path = config.REPO_ROOT / path
    return path


def load_snapshot(name: str) -> pd.DataFrame:
    """Load one frozen Phase 3 parquet contract by logical table name."""
    if name not in OUTPUT_FILES:
        valid = ", ".join(sorted(OUTPUT_FILES))
        raise ValueError(f"Unknown snapshot table {name!r}; expected one of: {valid}")
    return pd.read_parquet(snapshot_dir() / OUTPUT_FILES[name])
