"""Tests for Phase 3 payment-fraud features."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.features.multi_accounting import build_multi_accounting_features
from frauddet.features.payment import FEATURE_SPECS, build_payment_features
from frauddet.features.withdrawals import build_withdrawal_context


THRESHOLDS = {
    "pay_fast_window_minutes": 60,
    "pay_pct_withdrawn_threshold": 0.8,
    "pay_intervening_turnover_pct": 0.1,
    "pay_min_deposit_denominator": 10_000,
}


def _players(*player_keys):
    return pd.DataFrame(
        [
            [key, f"70000000{index}", "2026-06-01T08:00:00Z", None, None, None]
            for index, key in enumerate(player_keys, start=1)
        ],
        columns=[
            "player_key",
            "phone",
            "created_at",
            "nin_hash",
            "email_hash",
            "referred_by_key",
        ],
    )


def _money(rows=()):
    return pd.DataFrame(
        rows,
        columns=[
            "player_key",
            "txn_type",
            "transaction_id",
            "amount",
            "currency",
            "payment_method",
            "account_number",
            "final_status",
            "requested_at",
            "finalized_at",
            "recipient_normalized",
            "is_third_party_recipient",
            "is_money_in",
            "is_money_out",
        ],
    )


def _deposit(player, transaction_id, amount, finalized_at, status="completed"):
    return [
        player,
        "DEPOSIT",
        transaction_id,
        amount,
        "UGX",
        "mobile_money",
        f"acct-{player}",
        status,
        finalized_at,
        finalized_at,
        None,
        False,
        True,
        False,
    ]


def _withdrawal(
    player,
    transaction_id,
    amount,
    requested_at,
    *,
    status="completed",
    recipient=None,
    third_party=False,
):
    return [
        player,
        "WITHDRAWAL",
        transaction_id,
        amount,
        "UGX",
        "mobile_money",
        f"acct-{player}",
        status,
        requested_at,
        requested_at,
        recipient,
        third_party,
        False,
        status == "completed",
    ]


def _bets(rows=()):
    return pd.DataFrame(rows, columns=["player_key", "ticket_id", "stake", "created_at"])


def _build(players, money, bets):
    return build_payment_features(
        players=players,
        money=money,
        bets=bets,
        thresholds=THRESHOLDS,
        write_outputs=False,
    )


def _evidence(result, player_key, feature_name):
    row = result.feature_evidence[
        result.feature_evidence["player_key"].eq(player_key)
        & result.feature_evidence["feature_name"].eq(feature_name)
    ].iloc[0]
    return json.loads(row["feature_evidence"])


def test_pairing_uses_nearest_deposit_and_only_intervening_stake():
    players = _players("p1")
    money = _money(
        [
            _deposit("p1", "d-old", 100_000, "2026-06-01T09:00:00Z"),
            _deposit("p1", "d-near", 100_000, "2026-06-01T10:00:00Z"),
            _withdrawal("p1", "w1", 90_000, "2026-06-01T10:30:00Z"),
        ]
    )
    bets = _bets(
        [
            ["p1", "before", 40_000, "2026-06-01T09:30:00Z"],
            ["p1", "inside", 5_000, "2026-06-01T10:10:00Z"],
            ["p1", "after", 50_000, "2026-06-01T10:40:00Z"],
        ]
    )

    result = _build(players, money, bets)
    row = result.player_features.iloc[0]
    evidence = _evidence(result, "p1", "pay_deposit_then_exit_flag")[0]

    assert bool(row["pay_deposit_then_exit_flag"])
    assert row["pay_intervening_turnover_ratio"] == 0.05
    assert row["pay_min_minutes_deposit_to_withdrawal"] == 30
    assert evidence["deposit_id"] == "d-near"
    assert evidence["withdrawal_id"] == "w1"
    assert evidence["intervening_bet_ids"] == ["inside"]
    assert evidence["intervening_stake_sum"] == 5_000


def test_three_way_turnover_active_zero_is_value_inactive_is_casino_null():
    players = _players("active", "inactive")
    money = _money(
        [
            _deposit("active", "d-active", 100_000, "2026-06-01T10:00:00Z"),
            _withdrawal("active", "w-active", 90_000, "2026-06-01T10:30:00Z"),
            _deposit("inactive", "d-inactive", 100_000, "2026-06-01T10:00:00Z"),
            _withdrawal("inactive", "w-inactive", 90_000, "2026-06-01T10:30:00Z"),
        ]
    )
    bets = _bets([["active", "outside-window", 1_000, "2026-05-31T10:00:00Z"]])

    result = _build(players, money, bets)
    features = result.player_features.set_index("player_key")

    assert features.loc["active", "pay_intervening_turnover_ratio"] == 0.0
    assert bool(features.loc["active", "pay_deposit_then_exit_flag"])
    assert pd.isna(features.loc["inactive", "pay_intervening_turnover_ratio"])
    assert features.loc[
        "inactive", "pay_intervening_turnover_ratio__null_reason"
    ] == "casino_activity_not_observable"
    assert pd.isna(features.loc["inactive", "pay_deposit_then_exit_flag"])
    assert features.loc[
        "inactive", "pay_deposit_then_exit_flag__null_reason"
    ] == "casino_activity_not_observable"

    no_money = _build(_players("unobserved"), _money(), _bets())
    no_money_row = no_money.player_features.iloc[0]
    assert no_money_row["pay_deposit_then_exit_flag__null_reason"] == (
        "casino_activity_not_observable"
    )


def test_denominator_gate_and_real_zero_ratio_are_distinct():
    players = _players("low", "zero")
    money = _money(
        [
            _deposit("low", "d-low", 5_000, "2026-06-01T10:00:00Z"),
            _withdrawal("low", "w-low", 5_000, "2026-06-01T10:30:00Z"),
            _deposit("zero", "d-zero", 20_000, "2026-06-01T10:00:00Z"),
        ]
    )

    result = _build(players, money, _bets())
    features = result.player_features.set_index("player_key")

    assert pd.isna(features.loc["low", "pay_withdrawal_to_deposit_ratio"])
    assert features.loc[
        "low", "pay_withdrawal_to_deposit_ratio__null_reason"
    ] == "insufficient_deposit_denominator"
    assert features.loc["zero", "pay_withdrawal_to_deposit_ratio"] == 0.0
    assert pd.isna(features.loc["zero", "pay_withdrawal_to_deposit_ratio__null_reason"])

    no_deposit = _build(_players("none"), _money(), _bets())
    assert no_deposit.player_features.iloc[0][
        "pay_withdrawal_to_deposit_ratio__null_reason"
    ] == "no_completed_deposits"


def test_shared_withdrawal_context_feeds_ma_and_pay_recipient_features():
    players = _players("p1", "p2")
    recipient = "700999999"
    money = _money(
        [
            _withdrawal(
                "p1", "w1", 10_000, "2026-06-01T10:00:00Z", recipient=recipient, third_party=True
            ),
            _withdrawal(
                "p2", "w2", 20_000, "2026-06-01T10:05:00Z", recipient=recipient, third_party=True
            ),
        ]
    )
    context = build_withdrawal_context(players, money)
    ma_result = build_multi_accounting_features(
        players=players,
        money=money,
        logins=pd.DataFrame(columns=["player_key", "user_type", "fingerprint", "source_id"]),
        withdrawal_context=context,
        write_outputs=False,
    )
    pay_result = build_payment_features(
        players=players,
        money=money,
        bets=_bets(),
        withdrawal_context=context,
        thresholds=THRESHOLDS,
        write_outputs=False,
    )

    ma_features = ma_result.player_features.set_index("player_key")
    pay_features = pay_result.player_features.set_index("player_key")
    assert ma_features.loc["p1", "ma_withdrawal_recipient_shared_count"] == 1
    assert ma_features.loc["p2", "ma_withdrawal_recipient_shared_count"] == 1
    assert pay_features.loc["p1", "pay_third_party_withdrawal_count"] == 1
    assert pay_features.loc["p2", "pay_third_party_withdrawal_count"] == 1
    assert _evidence(pay_result, "p1", "pay_third_party_withdrawal_count") == [
        {"recipient_number": recipient, "withdrawal_id": "w1"}
    ]
    assert context.recipient.key_player_records[recipient] == {
        "p1": ("w1",),
        "p2": ("w2",),
    }


def test_every_nonzero_scoring_payment_feature_has_evidence():
    players = _players("p1")
    money = _money(
        [
            _deposit("p1", "d1", 100_000, "2026-06-01T10:00:00Z"),
            _withdrawal(
                "p1", "w1", 90_000, "2026-06-01T10:30:00Z", recipient="700999999", third_party=True
            ),
        ]
    )
    result = _build(
        players,
        money,
        _bets([["p1", "outside-window", 1_000, "2026-05-31T10:00:00Z"]]),
    )
    evidence = {
        (row.player_key, row.feature_name): json.loads(row.feature_evidence)
        for row in result.feature_evidence.itertuples(index=False)
    }
    values = result.player_features.iloc[0]
    for feature_name, spec in FEATURE_SPECS.items():
        value = values[feature_name]
        if spec.scoring_role == "scoring" and pd.notna(value) and bool(value):
            assert evidence[("p1", feature_name)]


def test_payment_features_reject_non_ugx_ledger_rows():
    players = _players("p1")
    money = _money([_deposit("p1", "d1", 100_000, "2026-06-01T10:00:00Z")])
    money.loc[:, "currency"] = "INR"

    with pytest.raises(ValueError, match="flattened UGX contract"):
        _build(players, money, _bets())
