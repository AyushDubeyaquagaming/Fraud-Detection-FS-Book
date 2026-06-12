"""Tests for Phase 2 flattening rules."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.flatten import (
    analyze_withdrawal_account_resolution,
    collapse_withdrawal_lifecycles,
    flatten_bonus,
    flatten_logins,
    flatten_money,
)
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


def test_deposit_finalized_at_uses_updated_at_with_created_fallback():
    deposits = [
        {
            "transactionId": "dep-updated",
            "userId": "0757575757",
            "amount": 1000,
            "currency": "UGX",
            "status": "completed",
            "createdAt": "2026-06-01T09:00:00Z",
            "updatedAt": "2026-06-01T09:05:00Z",
        },
        {
            "transactionId": "dep-created",
            "userId": "0757575757",
            "amount": 1000,
            "currency": "UGX",
            "status": "completed",
            "createdAt": "2026-06-01T09:01:00Z",
        },
    ]

    money = flatten_money(deposits, [], _mapper()).set_index("transaction_id")

    assert money.loc["dep-updated", "requested_at"] == pd.Timestamp("2026-06-01T09:00:00Z")
    assert money.loc["dep-updated", "finalized_at"] == pd.Timestamp("2026-06-01T09:05:00Z")
    assert money.loc["dep-created", "finalized_at"] == pd.Timestamp("2026-06-01T09:01:00Z")


def test_flatten_logins_marks_staff_and_fingerprint_sentinel():
    docs = [
        {
            "_id": "player-login",
            "loginId": "0757575757",
            "userType": "PLAYER",
            "operationType": "Login",
            "fingerprint": "a" * 64,
            "createdAt": "2026-06-01T10:00:00Z",
        },
        {
            "_id": "staff-login",
            "loginId": "adminmanager",
            "userType": "MANAGER",
            "operationType": "Login",
            "fingerprint": "not-found",
            "createdAt": "2026-06-01T10:01:00Z",
        },
    ]

    logins = flatten_logins(docs, _mapper()).set_index("source_id")

    assert logins.loc["player-login", "player_key"] == PLAYER_KEY
    assert logins.loc["player-login", "fingerprint"] == "a" * 64
    assert bool(logins.loc["player-login", "success"]) is True
    assert logins.loc["staff-login", "unjoined_class"] == "staff"
    assert pd.isna(logins.loc["staff-login", "fingerprint"])


def test_bonus_ref_kind_distinguishes_missing_and_unresolved():
    docs = [
        {
            "_id": "bonus-dep",
            "transactionType": "ALLOCATED",
            "userId": "0757575757",
            "amount": 10,
            "bonusTypeId": "First Deposit",
            "refTransId": "dep-1",
            "createdAt": "2026-06-01T10:00:00Z",
        },
        {
            "_id": "bonus-missing",
            "transactionType": "EXPIRED",
            "userId": "0757575757",
            "amount": 10,
            "bonusTypeId": "First Deposit",
            "createdAt": "2026-06-01T10:01:00Z",
        },
        {
            "_id": "bonus-unresolved",
            "transactionType": "RELEASE",
            "userId": "0757575757",
            "amount": 10,
            "bonusTypeId": "First Deposit",
            "refTransId": "not-found",
            "createdAt": "2026-06-01T10:02:00Z",
        },
    ]

    bonus = flatten_bonus(docs, _mapper(), {"dep-1"}, {"ticket-1"}).set_index("source_id")

    assert bonus.loc["bonus-dep", "ref_kind"] == "deposit"
    assert bonus.loc["bonus-missing", "ref_kind"] == "missing"
    assert bonus.loc["bonus-unresolved", "ref_kind"] == "unresolved"


def test_withdrawal_account_invariants_report_anomalies():
    analysis = analyze_withdrawal_account_resolution(
        [
            {
                "_id": "bad-wallet",
                "transactionId": "wd-wallet",
                "status": "completed",
                "toAccountId": "wallet-1",
            },
            {
                "_id": "bad-missing",
                "transactionId": "wd-missing",
                "status": "failed",
                "fromAccountId": None,
                "toAccountId": None,
            },
        ],
        [{"_id": "wallet-1"}],
        [],
    )

    assert len(analysis["anomalies"]) == 2
    assert not all(invariant["verified"] for invariant in analysis["invariants"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("flatten: all tests passed")
