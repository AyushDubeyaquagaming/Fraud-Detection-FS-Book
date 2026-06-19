"""Tests for Phase 2 flattening rules."""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.flatten import (
    analyze_withdrawal_account_resolution,
    collapse_withdrawal_lifecycles,
    flatten_bets,
    flatten_bonus,
    flatten_logins,
    flatten_money,
    flatten_players,
)
from frauddet.identity import IdentityMapper


PLAYER_KEY = "6a0ea9ff174ad3c431d9e16d"
TEST_HASH_SALT = "unit-test-identity-hash-salt"


def _mapper():
    return IdentityMapper.from_players(
        [{"_id": PLAYER_KEY, "username": "0757575757", "contactNo": "0757575757"}]
    )


def _set_test_hash_salt():
    os.environ["IDENTITY_HASH_SALT"] = TEST_HASH_SALT


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
    assert bool(money.loc["dep-manual", "is_money_in"]) is True
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


def test_flatten_players_opts_into_referred_by_player_id_resolution():
    _set_test_hash_salt()
    docs = [
        {
            "_id": "referrer",
            "username": "0757575757",
            "contactNo": "0757575757",
            "playerId": 10003905,
        },
        {
            "_id": "referred",
            "username": "0757676767",
            "contactNo": "0757676767",
            "referredBy": 10003905,
        },
    ]
    mapper = IdentityMapper.from_players(docs)

    players = flatten_players(docs, mapper).set_index("player_key")

    assert players.loc["referred", "referred_by_key"] == "referrer"


def test_flatten_players_hashes_identity_documents_without_raw_values():
    _set_test_hash_salt()
    docs = [
        {
            "_id": "p1",
            "username": "0757575757",
            "contactNo": "0757575757",
            "nin": " ab 123 cd ",
            "emailId": "Test@Example.COM ",
        },
        {
            "_id": "p2",
            "username": "0757676767",
            "contactNo": "0757676767",
            "nin": "AB123CD",
            "emailId": " test@example.com",
        },
        {
            "_id": "p3",
            "username": "0757777777",
            "contactNo": "0757777777",
            "nin": "DIFFERENT",
            "emailId": "other@example.com",
        },
        {
            "_id": "p4",
            "username": "0757878787",
            "contactNo": "0757878787",
            "nin": "   ",
            "emailId": "",
        },
    ]

    players = flatten_players(docs, IdentityMapper.from_players(docs)).set_index("player_key")

    assert players.loc["p1", "nin_hash"] == players.loc["p2", "nin_hash"]
    assert players.loc["p1", "email_hash"] == players.loc["p2", "email_hash"]
    assert players.loc["p1", "nin_hash"] != players.loc["p3", "nin_hash"]
    assert players.loc["p1", "email_hash"] != players.loc["p3", "email_hash"]
    assert pd.isna(players.loc["p4", "nin_hash"])
    assert pd.isna(players.loc["p4", "email_hash"])

    rendered_values = {
        str(value)
        for value in players[["nin_hash", "email_hash"]].to_numpy().ravel().tolist()
        if pd.notna(value)
    }
    assert all(len(value) == 64 for value in rendered_values)
    assert "AB123CD" not in rendered_values
    assert "test@example.com" not in rendered_values


def test_flatten_bets_relabels_mislabeled_inr_to_ugx_and_renames_settlement_time():
    bets = flatten_bets(
        [
            {
                "ticketId": "ticket-1",
                "loginId": "0757575757",
                "stake": 1000,
                "currency": "INR",
                "createdDate": "2026-06-01T10:00:00Z",
                "updatedAt": "2026-06-01T10:05:00Z",
            }
        ],
        _mapper(),
    )
    row = bets.iloc[0]

    assert row["currency"] == "UGX"
    assert row["source_currency"] == "INR"
    assert "settled_at_proxy" not in bets.columns
    assert row["settled_at"] == pd.Timestamp("2026-06-01T10:05:00Z")


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
