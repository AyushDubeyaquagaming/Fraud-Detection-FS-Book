"""Orchestrator for the reviewed Phase 3 feature groups."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..snapshot import load_snapshot
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
    """Build reviewed ma_ and pay_ groups from one frozen-snapshot load."""
    players = load_snapshot("players") if players is None else players.copy()
    money = load_snapshot("money") if money is None else money.copy()
    bets = load_snapshot("bets") if bets is None else bets.copy()
    logins = load_snapshot("logins") if logins is None else logins.copy()
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
    return combine_feature_build_results(
        ma_result,
        pay_result,
        output_dir=output_dir,
        write_outputs=write_outputs,
    )
