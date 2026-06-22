"""Tests for Phase 3 multi-accounting features."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.features.multi_accounting import (
    FEATURE_SPECS,
    build_multi_accounting_features,
)


def _players(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "player_key",
            "phone",
            "created_at",
            "nin_hash",
            "email_hash",
            "referred_by_key",
        ],
    )


def _logins(rows=()):
    return pd.DataFrame(
        rows,
        columns=["player_key", "user_type", "fingerprint", "source_id"],
    )


def _money(rows=()):
    return pd.DataFrame(
        rows,
        columns=[
            "player_key",
            "txn_type",
            "is_money_out",
            "recipient_normalized",
            "transaction_id",
        ],
    )


def _build(players, logins=None, money=None):
    return build_multi_accounting_features(
        players=players,
        logins=_logins() if logins is None else logins,
        money=_money() if money is None else money,
        write_outputs=False,
    )


def _evidence(result, player_key, feature_name):
    row = result.feature_evidence[
        result.feature_evidence["player_key"].eq(player_key)
        & result.feature_evidence["feature_name"].eq(feature_name)
    ].iloc[0]
    return json.loads(row["feature_evidence"])


def test_shared_hash_rolls_up_to_both_players_with_linked_evidence():
    players = _players(
        [
            ["p1", "700000001", "2026-06-01T10:00:00Z", "nin-shared", None, None],
            ["p2", "700000002", "2026-06-01T10:05:00Z", "nin-shared", None, None],
        ]
    )

    result = _build(players)
    features = result.player_features.set_index("player_key")

    assert features.loc["p1", "ma_nin_shared_account_count"] == 1
    assert features.loc["p2", "ma_nin_shared_account_count"] == 1
    assert _evidence(result, "p1", "ma_nin_shared_account_count") == [
        {
            "linked_source_record_ids": {"p2": []},
            "other_player_keys": ["p2"],
            "shared_key": "nin-shared",
            "shared_key_type": "nin_hash",
            "source_record_ids": [],
        }
    ]
    assert _evidence(result, "p2", "ma_nin_shared_account_count")[0][
        "other_player_keys"
    ] == ["p1"]


def test_linkage_is_one_hop_not_transitive():
    players = _players(
        [
            ["a", "700000001", "2026-06-01T10:00:00Z", "nin-ab", None, None],
            ["b", "700000002", "2026-06-01T10:05:00Z", "nin-ab", "email-bc", None],
            ["c", "700000003", "2026-06-01T10:10:00Z", "nin-c", "email-bc", None],
        ]
    )

    result = _build(players)
    features = result.player_features.set_index("player_key")

    assert features.loc["a", "ma_cocreated_linked_count"] == 1
    assert features.loc["b", "ma_cocreated_linked_count"] == 2
    assert features.loc["c", "ma_cocreated_linked_count"] == 1
    linked_from_a = [
        item["other_player_key"]
        for item in _evidence(result, "a", "ma_cocreated_linked_count")
    ]
    assert linked_from_a == ["b"]
    assert "c" not in linked_from_a


def test_null_contract_absent_fingerprint_is_null_absent_nin_is_zero():
    players = _players(
        [["p1", "700000001", "2026-06-01T10:00:00Z", None, None, None]]
    )

    result = _build(players)
    row = result.player_features.iloc[0]

    assert row["ma_nin_shared_account_count"] == 0
    assert pd.isna(row["ma_nin_shared_account_count__null_reason"])
    assert pd.isna(row["ma_device_shared_account_count"])
    assert row["ma_device_shared_account_count__null_reason"] == (
        "no_valid_fingerprint_logins"
    )
    assert pd.isna(row["ma_device_count"])
    assert row["ma_device_count__null_reason"] == "no_valid_fingerprint_logins"


def test_nonzero_scoring_features_always_have_evidence():
    fingerprint = "a" * 64
    players = _players(
        [
            ["p1", "700000001", "2026-06-01T10:00:00Z", "nin-shared", None, None],
            ["p2", "700000002", "2026-06-01T10:05:00Z", "nin-shared", None, "p1"],
        ]
    )
    logins = _logins(
        [
            ["p1", "PLAYER", fingerprint, "login-1"],
            ["p2", "PLAYER", fingerprint, "login-2"],
        ]
    )
    money = _money(
        [
            ["p1", "WITHDRAWAL", True, "700999999", "wd-1"],
            ["p2", "WITHDRAWAL", True, "700999999", "wd-2"],
        ]
    )

    result = _build(players, logins, money)
    evidence = {
        (row.player_key, row.feature_name): json.loads(row.feature_evidence)
        for row in result.feature_evidence.itertuples(index=False)
    }
    for row in result.player_features.itertuples(index=False):
        values = row._asdict()
        player_key = values["player_key"]
        for feature_name, spec in FEATURE_SPECS.items():
            value = values[feature_name]
            if spec.scoring_role == "scoring" and pd.notna(value) and bool(value):
                assert evidence[(player_key, feature_name)]

    referral_evidence = _evidence(result, "p2", "ma_referred_by_linked_account")
    assert referral_evidence[0]["referrer_player_key"] == "p1"
    assert referral_evidence[0]["corroborating_linkages"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("multi_accounting: all tests passed")
