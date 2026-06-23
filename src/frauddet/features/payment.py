"""Payment-fraud features built from frozen money and sportsbook records.

The payment group focuses on money movement: deposits, withdrawals, timing, and
recipient behavior. Some features also need sportsbook turnover to avoid saying
"no play" when casino play is simply not observable in v1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .. import config
from ..snapshot import load_snapshot
from .betting import sportsbook_active_players
from .output import FeatureBuildResult, FeatureSpec, materialize_feature_group
from .schema import FeatureResult, FeatureScoringRole, FeatureStrength
from .withdrawals import WithdrawalContext, build_withdrawal_context


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "pay_deposit_then_exit_flag": FeatureSpec("strong", "scoring", "bool"),
    "pay_intervening_turnover_ratio": FeatureSpec("strong", "scoring", "float"),
    "pay_min_minutes_deposit_to_withdrawal": FeatureSpec("strong", "scoring", "float"),
    "pay_third_party_withdrawal_flag": FeatureSpec("strong", "scoring", "bool"),
    "pay_third_party_withdrawal_count": FeatureSpec("strong", "scoring", "count"),
    "pay_withdrawal_to_deposit_ratio": FeatureSpec("moderate", "supporting", "float"),
    "pay_fast_withdrawal_count": FeatureSpec("moderate", "supporting", "count"),
    "pay_manual_reconciliation_count": FeatureSpec("moderate", "supporting", "count"),
    "pay_manual_reconciliation_ratio": FeatureSpec("moderate", "supporting", "float"),
    "pay_declined_withdrawal_count": FeatureSpec("moderate", "supporting", "count"),
    "pay_failed_withdrawal_count": FeatureSpec("weak", "context_only", "count"),
    "pay_net_money_flow": FeatureSpec("weak", "context_only", "float"),
    "pay_distinct_payment_methods": FeatureSpec("weak", "context_only", "count"),
    "pay_distinct_payment_accounts": FeatureSpec("weak", "context_only", "count"),
}


@dataclass(frozen=True)
class PaymentThresholds:
    """Config-driven placeholder thresholds for payment features."""

    fast_window_minutes: float
    pct_withdrawn_threshold: float
    intervening_turnover_pct: float
    min_deposit_denominator: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PaymentThresholds":
        return cls(
            fast_window_minutes=float(values["pay_fast_window_minutes"]),
            pct_withdrawn_threshold=float(values["pay_pct_withdrawn_threshold"]),
            intervening_turnover_pct=float(values["pay_intervening_turnover_pct"]),
            min_deposit_denominator=float(values["pay_min_deposit_denominator"]),
        )


@dataclass(frozen=True)
class DepositWithdrawalPair:
    """Nearest preceding deposit matched to one completed withdrawal."""

    deposit: dict[str, Any]
    withdrawal: dict[str, Any]
    gap_minutes: float


def build_payment_features(
    *,
    players: pd.DataFrame | None = None,
    money: pd.DataFrame | None = None,
    bets: pd.DataFrame | None = None,
    withdrawal_context: WithdrawalContext | None = None,
    thresholds: Mapping[str, Any] | None = None,
    output_dir: Path | None = None,
    write_outputs: bool = True,
) -> FeatureBuildResult:
    """Build exactly the Phase 3 v1 payment-fraud feature group.

    The builder reads frozen parquet by default. Tests can pass DataFrames
    directly, but feature work should not reach back into live Mongo.
    """
    players = load_snapshot("players") if players is None else players.copy()
    money = load_snapshot("money") if money is None else money.copy()
    bets = load_snapshot("bets") if bets is None else bets.copy()
    players = players[players["player_key"].notna()].copy()
    players["player_key"] = players["player_key"].astype(str)
    if players["player_key"].duplicated().any():
        raise ValueError("players input must contain one row per player_key.")

    population = set(players["player_key"])
    joined_money = money[
        money["player_key"].notna()
        & money["player_key"].astype(str).isin(population)
    ].copy()
    joined_money["player_key"] = joined_money["player_key"].astype(str)
    # Money ratios only make sense after the Phase 2 UGX contract is applied.
    # If a raw/mislabeled currency sneaks in, fail before producing features.
    ledger_rows = joined_money[
        joined_money["is_money_in"].fillna(False).astype(bool)
        | joined_money["is_money_out"].fillna(False).astype(bool)
    ]
    unexpected_currencies = set(ledger_rows["currency"].dropna().astype(str)) - {"UGX"}
    if unexpected_currencies:
        raise ValueError(
            "Payment features require the flattened UGX contract; found: "
            + ", ".join(sorted(unexpected_currencies))
        )
    joined_bets = bets[
        bets["player_key"].notna()
        & bets["player_key"].astype(str).isin(population)
    ].copy()
    joined_bets["player_key"] = joined_bets["player_key"].astype(str)
    withdrawals = withdrawal_context or build_withdrawal_context(players, money)
    threshold_values = thresholds or config.load_config()["thresholds"]
    payment_thresholds = PaymentThresholds.from_mapping(threshold_values)

    money_by_player = _group_records(joined_money)
    bets_by_player = _group_records(joined_bets)
    withdrawals_by_player = _group_records(withdrawals.withdrawals)
    completed_withdrawals_by_player = _group_records(withdrawals.completed)
    # This same helper is used by bet_game_type_concentration, so the payment
    # casino-null gate and betting context agree on who has sportsbook activity.
    sportsbook_active = sportsbook_active_players(joined_bets, population)

    def results_for(player_key: str) -> dict[str, FeatureResult]:
        return _player_results(
            player_key,
            money_by_player.get(player_key, []),
            bets_by_player.get(player_key, []),
            withdrawals_by_player.get(player_key, []),
            completed_withdrawals_by_player.get(player_key, []),
            player_key in sportsbook_active,
            payment_thresholds,
        )

    return materialize_feature_group(
        players["player_key"].tolist(),
        FEATURE_SPECS,
        results_for,
        output_dir=output_dir,
        write_outputs=write_outputs,
    )


def _player_results(
    player_key: str,
    money_rows: list[dict[str, Any]],
    bet_rows: list[dict[str, Any]],
    withdrawal_rows: list[dict[str, Any]],
    completed_withdrawals: list[dict[str, Any]],
    sportsbook_active: bool,
    thresholds: PaymentThresholds,
) -> dict[str, FeatureResult]:
    """Compute all pay_ feature results for one player."""
    deposits = [row for row in money_rows if bool(row.get("is_money_in"))]
    pairs = _pair_withdrawals_to_preceding_deposits(deposits, completed_withdrawals)
    fast_pairs = [
        pair for pair in pairs if pair.gap_minutes <= thresholds.fast_window_minutes
    ]
    # A trigger pair is a fast completed withdrawal that returns most of the
    # matched deposit. Turnover is checked separately after observability.
    trigger_pairs = [
        pair
        for pair in fast_pairs
        if _amount(pair.withdrawal) >= thresholds.pct_withdrawn_threshold * _amount(pair.deposit)
        and _amount(pair.deposit) > 0
    ]
    turnover_results = _turnover_results(
        deposits,
        completed_withdrawals,
        pairs,
        trigger_pairs,
        bet_rows,
        sportsbook_active,
        thresholds,
    )
    third_party = [
        row for row in completed_withdrawals if bool(row.get("is_third_party_recipient"))
    ]
    third_party_evidence = [
        {
            "withdrawal_id": str(row.get("transaction_id")),
            "recipient_number": _optional_string(row.get("recipient_normalized")),
        }
        for row in sorted(third_party, key=lambda row: str(row.get("transaction_id")))
    ]

    return {
        **turnover_results,
        "pay_min_minutes_deposit_to_withdrawal": _minimum_gap_result(
            deposits, completed_withdrawals, pairs
        ),
        "pay_third_party_withdrawal_flag": _completed_withdrawal_result(
            completed_withdrawals,
            bool(third_party),
            third_party_evidence,
            "strong",
            "scoring",
        ),
        "pay_third_party_withdrawal_count": _completed_withdrawal_result(
            completed_withdrawals,
            len(third_party),
            third_party_evidence,
            "strong",
            "scoring",
        ),
        "pay_withdrawal_to_deposit_ratio": _withdrawal_deposit_ratio(
            deposits, completed_withdrawals, thresholds.min_deposit_denominator
        ),
        # Timing-only signal: intentionally measurable for non-sportsbook
        # players. Phase 4 must keep it supporting-only and never use it as
        # standalone evidence against casino-only players.
        "pay_fast_withdrawal_count": _fast_withdrawal_result(
            deposits, completed_withdrawals, fast_pairs
        ),
        "pay_manual_reconciliation_count": _manual_reconciliation_result(
            deposits, ratio=False
        ),
        "pay_manual_reconciliation_ratio": _manual_reconciliation_result(
            deposits, ratio=True
        ),
        "pay_declined_withdrawal_count": _withdrawal_status_result(
            withdrawal_rows, "declined", "moderate", "supporting"
        ),
        "pay_failed_withdrawal_count": _withdrawal_status_result(
            withdrawal_rows, "failed", "weak", "context_only"
        ),
        "pay_net_money_flow": _net_money_flow_result(deposits, completed_withdrawals),
        "pay_distinct_payment_methods": _distinct_instrument_result(
            money_rows, "payment_method", "payment_methods"
        ),
        "pay_distinct_payment_accounts": _distinct_instrument_result(
            money_rows, "account_number", "payment_accounts"
        ),
    }


def _turnover_results(
    deposits: list[dict[str, Any]],
    withdrawals: list[dict[str, Any]],
    pairs: list[DepositWithdrawalPair],
    trigger_pairs: list[DepositWithdrawalPair],
    bets: list[dict[str, Any]],
    sportsbook_active: bool,
    thresholds: PaymentThresholds,
) -> dict[str, FeatureResult]:
    """Compute the turnover-dependent flagship features.

    The order matters: first decide whether sportsbook/casino turnover is
    observable, then decide if a deposit-withdrawal pair exists, then measure
    intervening stake for qualifying pairs.
    """
    if not _turnover_observable(sportsbook_active):
        return _null_turnover_results("casino_activity_not_observable")
    null_reason = _pair_null_reason(deposits, withdrawals, pairs)
    if null_reason:
        return _null_turnover_results(null_reason)
    if not trigger_pairs:
        return {
            "pay_deposit_then_exit_flag": FeatureResult(
                False, [], None, "strong", "scoring"
            ),
            "pay_intervening_turnover_ratio": FeatureResult(
                None,
                [],
                "no_triggering_deposit_withdrawal_pair",
                "strong",
                "scoring",
            ),
        }

    measured = [(_turnover_for_pair(pair, bets), pair) for pair in trigger_pairs]
    measured.sort(
        key=lambda item: (
            item[0][0],
            item[1].gap_minutes,
            str(item[1].deposit.get("transaction_id")),
            str(item[1].withdrawal.get("transaction_id")),
        )
    )
    (turnover_ratio, stake_sum, bet_ids), pair = measured[0]
    evidence = [
        {
            **_pair_evidence(pair),
            "pct_withdrawn": (_amount(pair.withdrawal) / _amount(pair.deposit)) * 100,
            "intervening_bet_ids": bet_ids,
            "intervening_stake_sum": stake_sum,
        }
    ]
    return {
        "pay_deposit_then_exit_flag": FeatureResult(
            turnover_ratio < thresholds.intervening_turnover_pct,
            evidence,
            None,
            "strong",
            "scoring",
        ),
        "pay_intervening_turnover_ratio": FeatureResult(
            turnover_ratio, evidence, None, "strong", "scoring"
        ),
    }


def _turnover_observable(
    sportsbook_active: bool,
    *,
    casino_zero_play_confirmed: bool = False,
) -> bool:
    """Return whether a player has observable play for turnover checks."""
    if sportsbook_active:
        return True
    # Future casino hook: player-level telemetry can make confirmed zero play
    # measurable. In v1 this branch is deliberately unreachable.
    if casino_zero_play_confirmed:
        return True
    return False


def _null_turnover_results(reason: str) -> dict[str, FeatureResult]:
    """Return the same null reason for both turnover-dependent features."""
    return {
        "pay_deposit_then_exit_flag": FeatureResult(
            None, [], reason, "strong", "scoring"
        ),
        "pay_intervening_turnover_ratio": FeatureResult(
            None, [], reason, "strong", "scoring"
        ),
    }


def _pair_withdrawals_to_preceding_deposits(
    deposits: list[dict[str, Any]],
    withdrawals: list[dict[str, Any]],
) -> list[DepositWithdrawalPair]:
    # Known production false-negative vector: v1 pairs each withdrawal to one
    # nearest preceding deposit. Splitting funds across several small deposits
    # before withdrawing the total can understate the denominator and evade the
    # flagship. Production calibration should evaluate cumulative-undrawn
    # deposit pairing or a comparable multi-deposit matching rule.
    timed_deposits = [
        (pd.to_datetime(row.get("finalized_at"), utc=True, errors="coerce"), row)
        for row in deposits
    ]
    timed_deposits = [(timestamp, row) for timestamp, row in timed_deposits if pd.notna(timestamp)]
    pairs: list[DepositWithdrawalPair] = []
    for withdrawal in withdrawals:
        requested = pd.to_datetime(
            withdrawal.get("requested_at"), utc=True, errors="coerce"
        )
        if pd.isna(requested):
            continue
        preceding = [(timestamp, row) for timestamp, row in timed_deposits if timestamp <= requested]
        if not preceding:
            continue
        finalized, deposit = max(
            preceding,
            key=lambda item: (item[0], str(item[1].get("transaction_id"))),
        )
        pairs.append(
            DepositWithdrawalPair(
                deposit=deposit,
                withdrawal=withdrawal,
                gap_minutes=(requested - finalized).total_seconds() / 60,
            )
        )
    return sorted(
        pairs,
        key=lambda pair: (
            pair.gap_minutes,
            str(pair.withdrawal.get("transaction_id")),
            str(pair.deposit.get("transaction_id")),
        ),
    )


def _turnover_for_pair(
    pair: DepositWithdrawalPair,
    bets: list[dict[str, Any]],
) -> tuple[float, float, list[str]]:
    """Stake between a matched deposit and withdrawal, divided by deposit size."""
    start = pd.to_datetime(pair.deposit.get("finalized_at"), utc=True, errors="coerce")
    end = pd.to_datetime(pair.withdrawal.get("requested_at"), utc=True, errors="coerce")
    intervening: list[dict[str, Any]] = []
    for bet in bets:
        created = pd.to_datetime(bet.get("created_at"), utc=True, errors="coerce")
        if pd.notna(created) and start <= created <= end:
            intervening.append(bet)
    stake_sum = sum(_number(row.get("stake")) for row in intervening)
    bet_ids = sorted(str(row.get("ticket_id")) for row in intervening)
    return stake_sum / _amount(pair.deposit), stake_sum, bet_ids


def _minimum_gap_result(
    deposits: list[dict[str, Any]],
    withdrawals: list[dict[str, Any]],
    pairs: list[DepositWithdrawalPair],
) -> FeatureResult:
    """Shortest completed-deposit to completed-withdrawal gap."""
    reason = _pair_null_reason(deposits, withdrawals, pairs)
    if reason:
        return FeatureResult(None, [], reason, "strong", "scoring")
    pair = min(pairs, key=lambda item: item.gap_minutes)
    return FeatureResult(
        pair.gap_minutes,
        [{**_pair_evidence(pair), "gap_minutes": pair.gap_minutes}],
        None,
        "strong",
        "scoring",
    )


def _fast_withdrawal_result(
    deposits: list[dict[str, Any]],
    withdrawals: list[dict[str, Any]],
    pairs: list[DepositWithdrawalPair],
) -> FeatureResult:
    """Count withdrawals inside the fast-exit window.

    This is a timing signal, not a turnover signal, so it remains measurable for
    casino-only players. Phase 4 must keep it supporting-only.
    """
    if not deposits:
        return FeatureResult(None, [], "no_completed_deposits", "moderate", "supporting")
    if not withdrawals:
        return FeatureResult(None, [], "no_completed_withdrawals", "moderate", "supporting")
    evidence = [_pair_evidence(pair) for pair in pairs]
    return FeatureResult(len(pairs), evidence, None, "moderate", "supporting")


def _completed_withdrawal_result(
    withdrawals: list[dict[str, Any]],
    value: int | bool,
    evidence: list[dict[str, Any]],
    strength: FeatureStrength,
    scoring_role: FeatureScoringRole,
) -> FeatureResult:
    """Shared helper for features that require completed withdrawals."""
    if not withdrawals:
        return FeatureResult(
            None, [], "no_completed_withdrawals", strength, scoring_role
        )
    return FeatureResult(value, evidence, None, strength, scoring_role)


def _withdrawal_deposit_ratio(
    deposits: list[dict[str, Any]],
    withdrawals: list[dict[str, Any]],
    denominator_floor: float,
) -> FeatureResult:
    """Completed money_out divided by completed money_in, with denominator gate."""
    if not deposits:
        return FeatureResult(
            None, [], "no_completed_deposits", "moderate", "supporting"
        )
    deposit_sum = sum(_amount(row) for row in deposits)
    withdrawal_sum = sum(_amount(row) for row in withdrawals)
    if deposit_sum < denominator_floor:
        return FeatureResult(
            None,
            [],
            "insufficient_deposit_denominator",
            "moderate",
            "supporting",
        )
    evidence = [
        {
            "reason_text": (
                f"completed money_out UGX {withdrawal_sum:g} / "
                f"completed money_in UGX {deposit_sum:g}"
            ),
            "money_in_sum_ugx": deposit_sum,
            "money_out_sum_ugx": withdrawal_sum,
        }
    ]
    return FeatureResult(
        withdrawal_sum / deposit_sum,
        evidence,
        None,
        "moderate",
        "supporting",
    )


def _manual_reconciliation_result(
    deposits: list[dict[str, Any]],
    *,
    ratio: bool,
) -> FeatureResult:
    """Count or ratio of credited manual-reconciliation deposits."""
    if not deposits:
        return FeatureResult(None, [], "no_completed_deposits", "moderate", "supporting")
    deposit_ids = sorted(
        str(row.get("transaction_id"))
        for row in deposits
        if str(row.get("final_status")) == "manual_reconciliation"
    )
    value: int | float = len(deposit_ids) / len(deposits) if ratio else len(deposit_ids)
    evidence = [{"deposit_ids": deposit_ids}] if deposit_ids else []
    return FeatureResult(value, evidence, None, "moderate", "supporting")


def _withdrawal_status_result(
    withdrawals: list[dict[str, Any]],
    status: str,
    strength: FeatureStrength,
    scoring_role: FeatureScoringRole,
) -> FeatureResult:
    """Count withdrawals ending in a given final status."""
    if not withdrawals:
        return FeatureResult(None, [], "no_withdrawals", strength, scoring_role)
    withdrawal_ids = sorted(
        str(row.get("transaction_id"))
        for row in withdrawals
        if str(row.get("final_status")) == status
    )
    evidence = [{"withdrawal_ids": withdrawal_ids}] if withdrawal_ids else []
    return FeatureResult(len(withdrawal_ids), evidence, None, strength, scoring_role)


def _net_money_flow_result(
    deposits: list[dict[str, Any]],
    withdrawals: list[dict[str, Any]],
) -> FeatureResult:
    """Completed deposits minus completed withdrawals."""
    money_in = sum(_amount(row) for row in deposits)
    money_out = sum(_amount(row) for row in withdrawals)
    return FeatureResult(
        money_in - money_out,
        [{"money_in_sum_ugx": money_in, "money_out_sum_ugx": money_out}],
        None,
        "weak",
        "context_only",
    )


def _distinct_instrument_result(
    money_rows: list[dict[str, Any]],
    column: str,
    evidence_name: str,
) -> FeatureResult:
    """Count distinct payment methods or accounts visible in money rows."""
    values = sorted(
        {
            value
            for row in money_rows
            if (value := _optional_string(row.get(column))) is not None
        }
    )
    evidence = [{evidence_name: values}] if values else []
    return FeatureResult(len(values), evidence, None, "weak", "context_only")


def _pair_null_reason(
    deposits: list[dict[str, Any]],
    withdrawals: list[dict[str, Any]],
    pairs: list[DepositWithdrawalPair],
) -> str | None:
    """Explain why deposit-withdrawal timing could not be measured."""
    if not deposits:
        return "no_completed_deposits"
    if not withdrawals:
        return "no_completed_withdrawals"
    if not pairs:
        return "no_subsequent_completed_withdrawal"
    return None


def _pair_evidence(pair: DepositWithdrawalPair) -> dict[str, Any]:
    return {
        "deposit_id": str(pair.deposit.get("transaction_id")),
        "withdrawal_id": str(pair.withdrawal.get("transaction_id")),
    }


def _group_records(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    return {
        str(player_key): group.to_dict("records")
        for player_key, group in frame.groupby("player_key", sort=False)
    }


def _amount(row: dict[str, Any]) -> float:
    return _number(row.get("amount"))


def _number(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _optional_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


# TODO(Phase 3+): pay_deposit_amount_structuring after production calibration.
# TODO(Phase 3+): pay_bank_* after bankDetails becomes populated and trustworthy.
