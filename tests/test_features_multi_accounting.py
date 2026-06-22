"""Tests for Phase 3 multi-accounting features."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.features.multi_accounting import (
    FEATURE_SPECS,
    build_multi_accounting_features,
)
from frauddet.features.schema import FeatureResult


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


def _build(players, logins=None, money=None, device_max_cardinality=None):
    return build_multi_accounting_features(
        players=players,
        logins=_logins() if logins is None else logins,
        money=_money() if money is None else money,
        write_outputs=False,
        device_max_cardinality=device_max_cardinality,
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
    assert row["ma_referral_fanout_count"] == 0
    assert _evidence(result, "p1", "ma_referral_fanout_count") == []
    assert _evidence(result, "p1", "ma_device_count") == []


def test_zero_allows_empty_evidence_but_nonzero_requires_evidence():
    zero = FeatureResult(0, [], None, "weak", "supporting")
    null = FeatureResult(
        None,
        [],
        "no_valid_fingerprint_logins",
        "weak",
        "context_only",
    )

    assert zero.feature_evidence == []
    assert null.feature_evidence == []
    with pytest.raises(ValueError, match="Non-zero feature values require"):
        FeatureResult(1, [], None, "strong", "scoring")


def test_device_cardinality_cap_blocks_links_and_corroboration_only_over_cap():
    over_cap = "a" * 64
    within_cap = "b" * 64
    players = _players(
        [
            ["p1", "700000001", "2026-06-01T10:00:00Z", None, None, None],
            ["p2", "700000002", "2026-06-01T10:05:00Z", None, None, "p1"],
            ["p3", "700000003", "2026-06-01T10:10:00Z", None, None, None],
            ["p4", "700000004", "2026-06-01T11:00:00Z", None, None, None],
            ["p5", "700000005", "2026-06-01T11:05:00Z", None, None, "p4"],
        ]
    )
    logins = _logins(
        [
            ["p1", "PLAYER", over_cap, "login-1"],
            ["p2", "PLAYER", over_cap, "login-2"],
            ["p3", "PLAYER", over_cap, "login-3"],
            ["p4", "PLAYER", within_cap, "login-4"],
            ["p5", "PLAYER", within_cap, "login-5"],
        ]
    )

    result = _build(players, logins, device_max_cardinality=2)
    features = result.player_features.set_index("player_key")

    for player_key in ["p1", "p2", "p3"]:
        assert features.loc[player_key, "ma_device_shared_account_count"] == 0
        assert features.loc[player_key, "ma_cocreated_linked_count"] == 0
        assert features.loc[player_key, "ma_device_count"] == 1
        assert _evidence(result, player_key, "ma_device_shared_account_count") == []
        assert over_cap not in json.dumps(
            _evidence(result, player_key, "ma_cocreated_linked_count")
        )
    assert not features.loc["p2", "ma_referred_by_linked_account"]
    assert _evidence(result, "p2", "ma_referred_by_linked_account") == []

    for player_key in ["p4", "p5"]:
        assert features.loc[player_key, "ma_device_shared_account_count"] == 1
        assert features.loc[player_key, "ma_cocreated_linked_count"] == 1
        assert within_cap in json.dumps(
            _evidence(result, player_key, "ma_device_shared_account_count")
        )
    assert features.loc["p5", "ma_referred_by_linked_account"]


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
