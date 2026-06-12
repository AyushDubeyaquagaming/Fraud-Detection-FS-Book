"""Read-only MongoDB access helpers.

Source collections are READ-ONLY (CLAUDE.md hard guardrail: never delete or
mutate source data). This module deliberately exposes only read operations
(list / count / sample / find). No insert/update/delete wrappers live here.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd
from pymongo import MongoClient
from pymongo.database import Database

from . import config


def get_client(**kwargs: Any) -> MongoClient:
    """Create a MongoClient from the URI in .env.

    serverSelectionTimeoutMS is kept short so a bad connection fails fast with a
    clear error instead of hanging.
    """
    kwargs.setdefault("serverSelectionTimeoutMS", 5000)
    return MongoClient(config.get_mongo_uri(), **kwargs)


@contextmanager
def get_database() -> Iterator[Database]:
    """Context manager yielding the data Database, closing the client on exit.

    Usage:
        with get_database() as db:
            ...
    """
    client = get_client()
    try:
        yield client[config.get_database_name()]
    finally:
        client.close()


def ping() -> dict[str, Any]:
    """Verify connectivity. Returns the server's ping response."""
    with get_database() as db:
        return db.command("ping")


def list_collections() -> list[str]:
    """Return sorted collection names in the data database."""
    with get_database() as db:
        return sorted(db.list_collection_names())


def count_docs(collection: str) -> int:
    """Exact document count for a collection."""
    with get_database() as db:
        return db[collection].count_documents({})


def count_all() -> dict[str, int]:
    """Document count for every collection in the database."""
    with get_database() as db:
        return {name: db[name].count_documents({}) for name in sorted(db.list_collection_names())}


def sample_docs(collection: str, n: int = 5) -> list[dict[str, Any]]:
    """Return up to n documents from a collection (read-only)."""
    with get_database() as db:
        return list(db[collection].find({}, limit=n))


def find_df(
    collection: str,
    query: dict[str, Any] | None = None,
    projection: dict[str, Any] | None = None,
    limit: int = 0,
) -> pd.DataFrame:
    """Run a read-only find() and return results as a DataFrame."""
    with get_database() as db:
        cursor = db[collection].find(query or {}, projection)
        if limit:
            cursor = cursor.limit(limit)
        return pd.DataFrame(list(cursor))
