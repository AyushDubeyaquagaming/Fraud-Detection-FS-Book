"""Orchestrator for the reviewed Phase 3 feature groups.

This is the single entry point for building the player feature table. It loads
the frozen snapshot once, shares common contexts, and writes one combined output.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..snapshot import load_snapshot
from .betting import build_betting_features
from .multi_accounting import build_multi_accounting_features
from .output import FeatureBuildResult, combine_feature_build_results
from .payment import build_payment_features
from .withdrawals import build_withdrawal_context


def build_phase3_features(
    *,
    players: pd.DataFrame | None = None,
    money: pd.DataFrame | None = None,
    bets: pd.DataFrame | None = None,
    logins: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    write_outputs: bool = True,
) -> FeatureBuildResult:
    """Build reviewed ma_, pay_, and bet_ groups from one frozen-snapshot load.

    Optional DataFrames are mainly for tests. When omitted, every input comes
    from `frauddet.snapshot.load_snapshot`, not live Mongo or mutable top-level
    parquet files.
    """
    players = load_snapshot("players") if players is None else players.copy()
    money = load_snapshot("money") if money is None else money.copy()
    bets = load_snapshot("bets") if bets is None else bets.copy()
    logins = load_snapshot("logins") if logins is None else logins.copy()
    # Withdrawal filtering and recipient linkage are shared between ma_ and pay_.
    # Building the context once prevents the groups from disagreeing.
    withdrawals = build_withdrawal_context(players, money)
    ma_result = build_multi_accounting_features(
        players=players,
        money=money,
        logins=logins,
        withdrawal_context=withdrawals,
        write_outputs=False,
    )
    pay_result = build_payment_features(
        players=players,
        money=money,
        bets=bets,
        withdrawal_context=withdrawals,
        write_outputs=False,
    )
    bet_result = build_betting_features(
        players=players,
        bets=bets,
        write_outputs=False,
    )
    return combine_feature_build_results(
        ma_result,
        pay_result,
        bet_result,
        output_dir=output_dir,
        write_outputs=write_outputs,
    )
