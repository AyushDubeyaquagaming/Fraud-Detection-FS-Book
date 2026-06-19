"""Configuration loading: merges committed config.yaml with secrets from .env.

Single source of truth for "where is the repo root", "what is the Mongo URI",
"what database / collections do we read". Everything else imports from here so
paths and names are defined once.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Repo root = two levels up from this file (src/frauddet/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"
ENV_PATH = REPO_ROOT / ".env"
DATA_DIR = REPO_ROOT / "data"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and cache config.yaml as a dict."""
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_mongo_uri() -> str:
    """Return the Mongo connection string from the environment / .env file.

    Raises a clear error if it is missing rather than failing deep inside pymongo.
    """
    load_dotenv(ENV_PATH)  # no-op if already loaded; safe to call repeatedly
    cfg = load_config()
    var = cfg["mongo"]["uri_env_var"]
    uri = os.environ.get(var)
    if not uri:
        raise RuntimeError(
            f"Mongo URI not found. Set {var} in {ENV_PATH} (copy from .env.example)."
        )
    return uri


def get_database_name() -> str:
    """Name of the data database to read from (NOT the auth database)."""
    return load_config()["mongo"]["database"]


def get_collection_names() -> dict[str, str | None]:
    """Configured source collection names verified in Phase 1."""
    return load_config()["collections"]


def get_identity_hash_salt() -> str:
    """Return the secret salt used for one-way identity-document hashes."""
    load_dotenv(ENV_PATH)
    var = load_config()["identity_hashing"]["salt_env_var"]
    salt = os.environ.get(var)
    if not salt:
        raise RuntimeError(
            f"Identity hash salt not found. Set {var} in {ENV_PATH}; never commit it."
        )
    return salt
