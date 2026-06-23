"""Tests for Phase 3 betting-anomaly features."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.features.betting import (  # noqa: E402
    FEATURE_SPECS,
    build_betting_features,
    sportsbook_active_players,
)
from frauddet.features.payment import build_payment_features  # noqa: E402


THRESHOLDS = {
    "bet_min_settled_bets_for_winrate": 3,
    "bet_min_bets_for_timing": 4,
    "bet_min_bets_for_volatility": 3,
    "bet_timing_cv_threshold": 0.2,
    "bet_win_rate_threshold": 0.7,
}

PAY_THRESHOLDS = {
    "pay_fast_window_minutes": 60,
    "pay_pct_withdrawn_threshold": 0.8,
    "pay_intervening_turnover_pct": 0.1,
    "pay_min_deposit_denominator": 10_000,
}


def _players(*player_keys):
    return pd.DataFrame(
        [[key] for key in player_keys],
        columns=["player_key"],
    )


def _bets(rows=()):
    return pd.DataFrame(
        rows,
        columns=[
            "player_key",
            "ticket_id",
            "game_type",
            "status",
            "result",
            "stake",
            "stake_bonus",
            "is_free_bet",
            "total_odds",
            "currency",
            "created_at",
        ],
    )


def _bet(
    player,
    ticket_id,
    created_at,
    *,
    game_type="Sports-book",
    status="SETTLED",
    result="LOSE",
    stake=100,
    stake_bonus=0,
    is_free_bet=False,
    total_odds=2.0,
):
    return [
        player,
        ticket_id,
        game_type,
        status,
        result,
        stake,
        stake_bonus,
        is_free_bet,
        total_odds,
        "UGX",
        created_at,
    ]


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


def _build(players, bets):
    return build_betting_features(
        players=players,
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


def test_volume_gates_below_threshold_null_at_threshold_measured():
    players = _players("below", "at")
    bets = _bets(
        [
            _bet("below", "b1", "2026-06-01T10:00:00Z", result="WIN"),
            _bet("below", "b2", "2026-06-01T10:10:00Z", result="LOSE"),
            _bet("at", "a1", "2026-06-01T10:00:00Z", result="WIN"),
            _bet("at", "a2", "2026-06-01T10:10:00Z", result="WIN"),
            _bet("at", "a3", "2026-06-01T10:20:00Z", result="LOSE"),
        ]
    )

    result = _build(players, bets)
    features = result.player_features.set_index("player_key")

    assert pd.isna(features.loc["below", "bet_win_rate_vs_volume"])
    assert features.loc["below", "bet_win_rate_vs_volume__null_reason"] == (
        "insufficient_settled_bets"
    )
    assert features.loc["at", "bet_win_rate_vs_volume"] == 2 / 3
    evidence = _evidence(result, "at", "bet_win_rate_vs_volume")[0]
    assert evidence["wins"] == 2
    assert evidence["settled_bets"] == 3
    assert evidence["winning_ticket_ids"] == ["a1", "a2"]


def test_timing_cv_computed_on_known_sequence():
    players = _players("p1")
    bets = _bets(
        [
            _bet("p1", "t1", "2026-06-01T10:00:00Z"),
            _bet("p1", "t2", "2026-06-01T10:01:00Z"),
            _bet("p1", "t3", "2026-06-01T10:03:00Z"),
            _bet("p1", "t4", "2026-06-01T10:06:00Z"),
        ]
    )

    result = _build(players, bets)
    row = result.player_features.iloc[0]

    assert round(float(row["bet_timing_regularity"]), 6) == round(0.408248290463863, 6)
    evidence = _evidence(result, "p1", "bet_timing_regularity")[0]
    assert evidence["ticket_ids"] == ["t1", "t2", "t3", "t4"]
    assert evidence["gap_seconds"] == [60.0, 120.0, 180.0]


def test_bonus_share_uses_stake_bonus_and_zero_is_measured():
    players = _players("bonus", "zero")
    bets = _bets(
        [
            _bet("bonus", "b1", "2026-06-01T10:00:00Z", stake=100, stake_bonus=25),
            _bet("bonus", "b2", "2026-06-01T10:10:00Z", stake=300, stake_bonus=0),
            _bet("zero", "z1", "2026-06-01T10:00:00Z", stake=200, stake_bonus=0),
        ]
    )

    result = _build(players, bets)
    features = result.player_features.set_index("player_key")

    assert features.loc["bonus", "bet_bonus_funded_stake_share"] == 25 / 400
    assert _evidence(result, "bonus", "bet_bonus_funded_stake_share")[0][
        "bonus_staked_ticket_ids"
    ] == ["b1"]
    assert features.loc["zero", "bet_bonus_funded_stake_share"] == 0.0
    assert _evidence(result, "zero", "bet_bonus_funded_stake_share") == []


def test_game_type_concentration_sums_to_one_and_matches_pay_sportsbook_active():
    players = _players("p1", "p2", "none")
    bets = _bets(
        [
            _bet("p1", "s1", "2026-06-01T10:00:00Z", stake=75),
            _bet("p1", "c1", "2026-06-01T10:05:00Z", game_type="Casino", stake=25),
            _bet("p2", "s2", "2026-06-01T10:00:00Z", stake=50),
        ]
    )

    result = _build(players, bets)
    features = result.player_features.set_index("player_key")
    evidence = _evidence(result, "p1", "bet_game_type_concentration")[0]
    shares = [item["stake_share"] for item in evidence["per_game_type"]]

    assert features.loc["p1", "bet_game_type_concentration"] == 0.75
    assert round(sum(shares), 10) == 1.0
    assert sportsbook_active_players(bets, {"p1", "p2", "none"}) == frozenset({"p1", "p2"})

    pay = build_payment_features(
        players=players,
        money=_money(),
        bets=bets,
        thresholds=PAY_THRESHOLDS,
        write_outputs=False,
    )
    pay_features = pay.player_features.set_index("player_key")
    pay_active = set(
        pay_features[
            pay_features["pay_deposit_then_exit_flag__null_reason"].ne(
                "casino_activity_not_observable"
            )
        ].index
    )
    assert pay_active == {"p1", "p2"}


def test_no_bets_null_vs_zero_contract():
    result = _build(_players("none"), _bets())
    row = result.player_features.iloc[0]

    assert row["bet_count"] == 0
    assert row["bet_active_days"] == 0
    assert pd.isna(row["bet_win_rate_vs_volume"])
    assert row["bet_win_rate_vs_volume__null_reason"] == "no_bets"
    assert pd.isna(row["bet_bonus_funded_stake_share"])
    assert row["bet_bonus_funded_stake_share__null_reason"] == "no_bets"
    assert pd.isna(row["bet_void_rate"])
    assert row["bet_void_rate__null_reason"] == "no_bets"
    assert _evidence(result, "none", "bet_count") == []


def test_every_nonzero_scoring_betting_feature_has_evidence():
    players = _players("p1")
    bets = _bets(
        [
            _bet("p1", "t1", "2026-06-01T10:00:00Z", result="WIN"),
            _bet("p1", "t2", "2026-06-01T10:01:00Z", result="WIN"),
            _bet("p1", "t3", "2026-06-01T10:02:00Z", result="LOSE"),
            _bet("p1", "t4", "2026-06-01T10:03:00Z", result="LOSE"),
        ]
    )
    result = _build(players, bets)
    evidence = {
        (row.player_key, row.feature_name): json.loads(row.feature_evidence)
        for row in result.feature_evidence.itertuples(index=False)
    }
    values = result.player_features.iloc[0]
    for feature_name, spec in FEATURE_SPECS.items():
        value = values[feature_name]
        if spec.scoring_role == "scoring" and pd.notna(value) and bool(value):
            assert evidence[("p1", feature_name)]
