"""Betting-anomaly features built from frozen bet records.

These are statistical signals, so most of them are volume-gated and expected to
be dormant on the small dev snapshot. The code is built for production
calibration: compute per game type, keep reviewer evidence, and return honest
nulls when a player has too little betting history.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .. import config
from ..snapshot import load_snapshot
from .output import FeatureBuildResult, FeatureSpec, materialize_feature_group
from .schema import FeatureResult


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "bet_win_rate_vs_volume": FeatureSpec("strong", "scoring", "float"),
    "bet_timing_regularity": FeatureSpec("strong", "scoring", "float"),
    "bet_stake_volatility": FeatureSpec("moderate", "supporting", "float"),
    "bet_bonus_funded_stake_share": FeatureSpec("moderate", "supporting", "float"),
    "bet_game_type_concentration": FeatureSpec("moderate", "context_only", "float"),
    "bet_avg_odds": FeatureSpec("weak", "context_only", "float"),
    "bet_odds_profile": FeatureSpec("weak", "context_only", "float"),
    "bet_count": FeatureSpec("weak", "context_only", "count"),
    "bet_active_days": FeatureSpec("weak", "context_only", "count"),
    "bet_void_rate": FeatureSpec("weak", "context_only", "float"),
}


@dataclass(frozen=True)
class BettingThresholds:
    """Config-driven placeholder thresholds for betting features."""

    min_settled_bets_for_winrate: int
    min_bets_for_timing: int
    min_bets_for_volatility: int
    timing_cv_threshold: float
    win_rate_threshold: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BettingThresholds":
        return cls(
            min_settled_bets_for_winrate=int(values["bet_min_settled_bets_for_winrate"]),
            min_bets_for_timing=int(values["bet_min_bets_for_timing"]),
            min_bets_for_volatility=int(values["bet_min_bets_for_volatility"]),
            timing_cv_threshold=float(values["bet_timing_cv_threshold"]),
            win_rate_threshold=float(values["bet_win_rate_threshold"]),
        )


def build_betting_features(
    *,
    players: pd.DataFrame | None = None,
    bets: pd.DataFrame | None = None,
    thresholds: Mapping[str, Any] | None = None,
    output_dir: Path | None = None,
    write_outputs: bool = True,
) -> FeatureBuildResult:
    """Build exactly the Phase 3 v1 betting-anomaly feature group.

    The scalar output stays one column per feature. When a feature is computed
    per game type, the scalar is the dominant or most suspicious per-game value
    and the full per-game breakdown is stored in evidence.
    """
    players = load_snapshot("players") if players is None else players.copy()
    bets = load_snapshot("bets") if bets is None else bets.copy()
    players = players[players["player_key"].notna()].copy()
    players["player_key"] = players["player_key"].astype(str)
    if players["player_key"].duplicated().any():
        raise ValueError("players input must contain one row per player_key.")

    population = set(players["player_key"])
    joined_bets = bets[
        bets["player_key"].notna()
        & bets["player_key"].astype(str).isin(population)
    ].copy()
    joined_bets["player_key"] = joined_bets["player_key"].astype(str)
    # Betting ratios assume the Phase 2 currency relabel has already happened.
    # This protects feature work from accidentally reading raw source rows.
    unexpected_currencies = set(joined_bets["currency"].dropna().astype(str)) - {"UGX"}
    if unexpected_currencies:
        raise ValueError(
            "Betting features require the flattened UGX contract; found: "
            + ", ".join(sorted(unexpected_currencies))
        )

    threshold_values = thresholds or config.load_config()["thresholds"]
    betting_thresholds = BettingThresholds.from_mapping(threshold_values)
    bets_by_player = _group_records(joined_bets)

    def results_for(player_key: str) -> dict[str, FeatureResult]:
        return _player_results(bets_by_player.get(player_key, []), betting_thresholds)

    return materialize_feature_group(
        players["player_key"].tolist(),
        FEATURE_SPECS,
        results_for,
        output_dir=output_dir,
        write_outputs=write_outputs,
    )


def sportsbook_active_players(
    bets: pd.DataFrame,
    population: set[str] | frozenset[str] | None = None,
) -> frozenset[str]:
    """Return players with any joined sportsbook bet in the flattened contract.

    Payment turnover logic uses this same helper so `pay_` casino-null behavior
    and `bet_game_type_concentration` use the same definition of observable
    sportsbook activity.
    """
    if bets.empty or "player_key" not in bets:
        return frozenset()
    joined = bets[bets["player_key"].notna()].copy()
    joined["player_key"] = joined["player_key"].astype(str)
    if population is not None:
        joined = joined[joined["player_key"].isin(set(population))]
    return frozenset(joined["player_key"].unique())


def _player_results(
    bet_rows: list[dict[str, Any]],
    thresholds: BettingThresholds,
) -> dict[str, FeatureResult]:
    """Compute all bet_ feature results for one player."""
    groups = _groups_by_game_type(bet_rows)
    return {
        "bet_win_rate_vs_volume": _win_rate_vs_volume(groups, thresholds),
        "bet_timing_regularity": _timing_regularity(groups, thresholds),
        "bet_stake_volatility": _stake_volatility(groups, thresholds),
        "bet_bonus_funded_stake_share": _bonus_funded_stake_share(bet_rows),
        "bet_game_type_concentration": _game_type_concentration(groups),
        "bet_avg_odds": _avg_odds(bet_rows),
        "bet_odds_profile": _odds_profile(bet_rows),
        "bet_count": _bet_count(bet_rows),
        "bet_active_days": _active_days(bet_rows),
        "bet_void_rate": _void_rate(bet_rows),
    }


def _win_rate_vs_volume(
    groups: dict[str, list[dict[str, Any]]],
    thresholds: BettingThresholds,
) -> FeatureResult:
    """Highest per-game settled win rate after the minimum-volume gate."""
    if not groups:
        return FeatureResult(None, [], "no_bets", "strong", "scoring")
    measured: list[dict[str, Any]] = []
    for game_type, rows in sorted(groups.items()):
        settled = [row for row in rows if str(row.get("status")) == "SETTLED"]
        if len(settled) < thresholds.min_settled_bets_for_winrate:
            continue
        wins = [row for row in settled if str(row.get("result")) == "WIN"]
        measured.append(
            {
                "game_type": game_type,
                "win_rate": len(wins) / len(settled),
                "winning_ticket_ids": _ticket_ids(wins),
                "wins": len(wins),
                "settled_bets": len(settled),
                "reason_text": f"{len(wins)} wins / {len(settled)} settled bets",
            }
        )
    if not measured:
        return FeatureResult(
            None,
            [],
            "insufficient_settled_bets",
            "strong",
            "scoring",
        )
    evidence = sorted(measured, key=lambda item: (-item["win_rate"], item["game_type"]))
    return FeatureResult(evidence[0]["win_rate"], evidence, None, "strong", "scoring")


def _timing_regularity(
    groups: dict[str, list[dict[str, Any]]],
    thresholds: BettingThresholds,
) -> FeatureResult:
    """Lowest per-game coefficient of variation for inter-bet gaps."""
    if not groups:
        return FeatureResult(None, [], "no_bets", "strong", "scoring")
    measured: list[dict[str, Any]] = []
    for game_type, rows in sorted(groups.items()):
        ordered = _ordered_by_created_at(rows)
        if len(ordered) < thresholds.min_bets_for_timing:
            continue
        gaps = _inter_bet_gap_seconds(ordered)
        cv = _coefficient_of_variation(gaps)
        measured.append(
            {
                "game_type": game_type,
                "coefficient_of_variation": cv,
                "ticket_ids": _ticket_ids(ordered),
                "timestamps": [_timestamp(row.get("created_at")) for row in ordered],
                "gap_seconds": gaps,
            }
        )
    if not measured:
        return FeatureResult(
            None,
            [],
            "insufficient_bets_for_timing",
            "strong",
            "scoring",
        )
    evidence = sorted(
        measured,
        key=lambda item: (item["coefficient_of_variation"], item["game_type"]),
    )
    return FeatureResult(
        evidence[0]["coefficient_of_variation"],
        evidence,
        None,
        "strong",
        "scoring",
    )


def _stake_volatility(
    groups: dict[str, list[dict[str, Any]]],
    thresholds: BettingThresholds,
) -> FeatureResult:
    """Largest per-game stake coefficient of variation after the volume gate."""
    if not groups:
        return FeatureResult(None, [], "no_bets", "moderate", "supporting")
    measured: list[dict[str, Any]] = []
    for game_type, rows in sorted(groups.items()):
        if len(rows) < thresholds.min_bets_for_volatility:
            continue
        stakes = [_number(row.get("stake")) for row in rows]
        mean = float(pd.Series(stakes, dtype="float64").mean())
        std = float(pd.Series(stakes, dtype="float64").std(ddof=0))
        value = std / mean if mean > 0 else 0.0
        ordered_by_stake = sorted(
            rows,
            key=lambda row: (_number(row.get("stake")), str(row.get("ticket_id"))),
        )
        measured.append(
            {
                "game_type": game_type,
                "stake_cv": value,
                "stake_mean": mean,
                "stake_std": std,
                "min_stake": min(stakes),
                "max_stake": max(stakes),
                "extreme_ticket_ids": _ticket_ids(
                    [ordered_by_stake[0], ordered_by_stake[-1]]
                ),
            }
        )
    if not measured:
        return FeatureResult(
            None,
            [],
            "insufficient_bets_for_volatility",
            "moderate",
            "supporting",
        )
    evidence = sorted(measured, key=lambda item: (-item["stake_cv"], item["game_type"]))
    return FeatureResult(evidence[0]["stake_cv"], evidence, None, "moderate", "supporting")


def _bonus_funded_stake_share(rows: list[dict[str, Any]]) -> FeatureResult:
    """Share of total stake funded by bonus stake or free bets."""
    if not rows:
        return FeatureResult(None, [], "no_bets", "moderate", "supporting")
    stake_sum = sum(_number(row.get("stake")) for row in rows)
    bonus_sum = sum(_number(row.get("stake_bonus")) for row in rows)
    bonus_rows = [
        row
        for row in rows
        if _number(row.get("stake_bonus")) > 0 or bool(row.get("is_free_bet"))
    ]
    share = bonus_sum / stake_sum if stake_sum > 0 else 0.0
    evidence = (
        [
            {
                "bonus_staked_ticket_ids": _ticket_ids(bonus_rows),
                "bonus_stake_sum": bonus_sum,
                "free_bet_count": sum(bool(row.get("is_free_bet")) for row in rows),
                "total_stake": stake_sum,
            }
        ]
        if bonus_rows
        else []
    )
    return FeatureResult(share, evidence, None, "moderate", "supporting")


def _game_type_concentration(groups: dict[str, list[dict[str, Any]]]) -> FeatureResult:
    """Dominant game-type stake share plus full per-type evidence."""
    if not groups:
        return FeatureResult(None, [], "no_bets", "moderate", "context_only")
    stake_by_type = {
        game_type: sum(_number(row.get("stake")) for row in rows)
        for game_type, rows in groups.items()
    }
    total_stake = sum(stake_by_type.values())
    if total_stake <= 0:
        share_by_type = {game_type: 0.0 for game_type in stake_by_type}
    else:
        share_by_type = {
            game_type: stake_sum / total_stake
            for game_type, stake_sum in stake_by_type.items()
        }
    dominant_game_type, dominant_share = max(
        share_by_type.items(), key=lambda item: (item[1], item[0])
    )
    evidence = [
        {
            "dominant_game_type": dominant_game_type,
            "dominant_share": dominant_share,
            "per_game_type": [
                {
                    "game_type": game_type,
                    "bet_count": len(groups[game_type]),
                    "stake_sum": stake_by_type[game_type],
                    "stake_share": share_by_type[game_type],
                    "ticket_ids": _ticket_ids(groups[game_type]),
                }
                for game_type in sorted(groups)
            ],
        }
    ]
    return FeatureResult(dominant_share, evidence, None, "moderate", "context_only")


def _avg_odds(rows: list[dict[str, Any]]) -> FeatureResult:
    """Mean total odds for players with any observable bets."""
    odds = [
        _number(row.get("total_odds"))
        for row in rows
        if _has_number(row.get("total_odds"))
    ]
    if not odds:
        return FeatureResult(None, [], "no_bets", "weak", "context_only")
    return FeatureResult(
        float(pd.Series(odds, dtype="float64").mean()),
        [{"ticket_ids": _ticket_ids(rows), "odds_count": len(odds)}],
        None,
        "weak",
        "context_only",
    )


def _odds_profile(rows: list[dict[str, Any]]) -> FeatureResult:
    """Spread of total odds as a context-only betting profile."""
    odds_rows = [row for row in rows if _has_number(row.get("total_odds"))]
    if not odds_rows:
        return FeatureResult(None, [], "no_bets", "weak", "context_only")
    odds = [_number(row.get("total_odds")) for row in odds_rows]
    spread = float(pd.Series(odds, dtype="float64").std(ddof=0))
    evidence = [
        {
            "ticket_ids": _ticket_ids(odds_rows),
            "odds_mean": float(pd.Series(odds, dtype="float64").mean()),
            "odds_std": spread,
            "odds_min": min(odds),
            "odds_max": max(odds),
        }
    ]
    return FeatureResult(spread, evidence, None, "weak", "context_only")


def _bet_count(rows: list[dict[str, Any]]) -> FeatureResult:
    """Raw bet count; zero is a real measured value."""
    evidence = [{"ticket_ids": _ticket_ids(rows)}] if rows else []
    return FeatureResult(len(rows), evidence, None, "weak", "context_only")


def _active_days(rows: list[dict[str, Any]]) -> FeatureResult:
    """Number of UTC calendar days with at least one bet."""
    days = sorted(
        {
            timestamp.date().isoformat()
            for row in rows
            if pd.notna(
                timestamp := pd.to_datetime(
                    row.get("created_at"), utc=True, errors="coerce"
                )
            )
        }
    )
    evidence = [{"active_dates": days, "ticket_ids": _ticket_ids(rows)}] if days else []
    return FeatureResult(len(days), evidence, None, "weak", "context_only")


def _void_rate(rows: list[dict[str, Any]]) -> FeatureResult:
    """Share of bets with VOID status."""
    if not rows:
        return FeatureResult(None, [], "no_bets", "weak", "context_only")
    void_rows = [row for row in rows if str(row.get("status")) == "VOID"]
    evidence = [{"void_ticket_ids": _ticket_ids(void_rows)}] if void_rows else []
    return FeatureResult(len(void_rows) / len(rows), evidence, None, "weak", "context_only")


def _groups_by_game_type(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by game_type, using 'unknown' only for missing labels."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        game_type = _optional_string(row.get("game_type")) or "unknown"
        groups.setdefault(game_type, []).append(row)
    return groups


def _ordered_by_created_at(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            pd.to_datetime(row.get("created_at"), utc=True, errors="coerce"),
            str(row.get("ticket_id")),
        ),
    )


def _inter_bet_gap_seconds(rows: list[dict[str, Any]]) -> list[float]:
    timestamps = [
        pd.to_datetime(row.get("created_at"), utc=True, errors="coerce")
        for row in rows
    ]
    return [
        (later - earlier).total_seconds()
        for earlier, later in zip(timestamps, timestamps[1:])
        if pd.notna(earlier) and pd.notna(later)
    ]


def _coefficient_of_variation(values: list[float]) -> float:
    """Population standard deviation divided by mean, with safe zero handling."""
    if not values:
        return 0.0
    series = pd.Series(values, dtype="float64")
    mean = float(series.mean())
    if mean == 0:
        return 0.0
    return float(series.std(ddof=0) / mean)


def _group_records(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    return {
        str(player_key): group.to_dict("records")
        for player_key, group in frame.groupby("player_key", sort=False)
    }


def _ticket_ids(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(str(row.get("ticket_id")) for row in rows)


def _timestamp(value: Any) -> str | None:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


def _has_number(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _number(value: Any) -> float:
    if not _has_number(value):
        return 0.0
    return float(value)


def _optional_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


# TODO(Phase 3+): casino game-exploitation features after per-round logging exists.
