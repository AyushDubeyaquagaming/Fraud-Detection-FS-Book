"""Tests for Phase 2 flattening rules."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.flatten import collapse_withdrawal_lifecycles, flatten_money
from frauddet.identity import IdentityMapper


PLAYER_KEY = "6a0ea9ff174ad3c431d9e16d"


def _mapper():
    return IdentityMapper.from_players(
        [{"_id": PLAYER_KEY, "username": "0757575757", "contactNo": "0757575757"}]
    )


def test_withdrawal_lifecycle_collapse_keeps_most_advanced_status():
    docs = [
        {
            "_id": "a",
            "transactionId": "wd-1",
            "status": "initiated",
            "createdAt": "2026-06-01T10:00:00Z",
            "updatedAt": "2026-06-01T10:00:01Z",
        },
        {
            "_id": "b",
            "transactionId": "wd-1",
            "status": "failed",
            "createdAt": "2026-06-01T10:02:00Z",
            "updatedAt": "2026-06-01T10:02:01Z",
        },
        {
            "_id": "c",
            "transactionId": "wd-1",
            "status": "declined",
            "createdAt": "2026-06-01T10:03:00Z",
            "updatedAt": "2026-06-01T10:03:01Z",
        },
    ]

    [group] = collapse_withdrawal_lifecycles(docs)

    assert group["kept_doc"]["status"] == "declined"
    assert group["requested_at"] == pd.Timestamp("2026-06-01T10:00:00Z")


def test_withdrawal_lifecycle_finalized_at_comes_from_kept_doc():
    withdrawals = [
        {
            "_id": "a",
            "transactionId": "wd-2",
            "userId": "0757575757",
            "amount": 1000,
            "currency": "UGX",
            "status": "initiated",
            "createdAt": "2026-06-01T10:00:00Z",
            "updatedAt": "2026-06-01T10:00:01Z",
        },
        {
            "_id": "b",
            "transactionId": "wd-2",
            "userId": "0757575757",
            "amount": 1000,
            "currency": "UGX",
            "status": "completed",
            "createdAt": "2026-06-01T10:05:00Z",
            "updatedAt": "2026-06-01T10:05:05Z",
        },
    ]

    money = flatten_money([], withdrawals, _mapper())
    row = money.iloc[0]

    assert row["final_status"] == "completed"
    assert row["requested_at"] == pd.Timestamp("2026-06-01T10:00:00Z")
    assert row["finalized_at"] == pd.Timestamp("2026-06-01T10:05:05Z")


def test_money_status_flags_match_config_whitelists():
    deposits = [
        {
            "transactionId": "dep-completed",
            "userId": "0757575757",
            "amount": 1000,
            "currency": "UGX",
            "status": "completed",
            "createdAt": "2026-06-01T09:00:00Z",
        },
        {
            "transactionId": "dep-manual",
            "userId": "0757575757",
            "amount": 1000,
            "currency": "UGX",
            "status": "manual_reconciliation",
            "createdAt": "2026-06-01T09:01:00Z",
        },
    ]
    withdrawals = [
        {
            "_id": "wd-completed",
            "transactionId": "wd-completed",
            "userId": "0757575757",
            "amount": 500,
            "currency": "UGX",
            "status": "completed",
            "createdAt": "2026-06-01T10:00:00Z",
            "updatedAt": "2026-06-01T10:00:01Z",
        },
        {
            "_id": "wd-pending",
            "transactionId": "wd-pending",
            "userId": "0757575757",
            "amount": 500,
            "currency": "UGX",
            "status": "pending",
            "createdAt": "2026-06-01T10:01:00Z",
            "updatedAt": "2026-06-01T10:01:01Z",
        },
    ]

    money = flatten_money(deposits, withdrawals, _mapper()).set_index("transaction_id")

    assert bool(money.loc["dep-completed", "is_money_in"]) is True
    assert bool(money.loc["dep-manual", "is_money_in"]) is False
    assert bool(money.loc["wd-completed", "is_money_out"]) is True
    assert bool(money.loc["wd-pending", "is_pending_withdrawal"]) is True
    assert bool(money.loc["wd-pending", "is_money_out"]) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("flatten: all tests passed")
