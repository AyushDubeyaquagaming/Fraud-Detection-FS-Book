# %% [markdown]
# # 00 — Phase 0 smoke test
# **Purpose:** prove the environment can reach MongoDB read-only.
# **Inputs:** MONGODB_URI (.env), config.yaml.
# **Outputs:** printed connection status, collection list, per-collection counts.
# Run as a script (`python notebooks/00_smoke_test.py`) or cell-by-cell.

# %%
import os
import sys
from pathlib import Path

# Make src/ importable when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet import config, io_mongo


def run_smoke_check() -> None:
    """Ping Mongo and print read-only collection counts."""
    print("Repo root :", config.REPO_ROOT)
    print("Database  :", config.get_database_name())
    print("Ping      :", io_mongo.ping())

    counts = io_mongo.count_all()
    print(f"\n{len(counts)} collections in '{config.get_database_name()}':\n")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<32} {n:>8,d} docs")


def test_mongo_smoke() -> None:
    """Run under pytest only when explicitly enabled."""
    from dotenv import load_dotenv
    import pytest

    load_dotenv(config.ENV_PATH)
    if os.environ.get("RUN_MONGO_SMOKE_TEST") != "1":
        pytest.skip("Set RUN_MONGO_SMOKE_TEST=1 to run the live Mongo smoke test.")
    run_smoke_check()


if __name__ == "__main__":
    run_smoke_check()
