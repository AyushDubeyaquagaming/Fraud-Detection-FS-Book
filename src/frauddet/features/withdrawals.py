"""Shared completed-withdrawal population and recipient linkage."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .linkage import LinkageIndex, build_frame_linkage


@dataclass(frozen=True)
class WithdrawalContext:
    withdrawals: pd.DataFrame
    completed: pd.DataFrame
    recipient: LinkageIndex
    withdrawal_players: frozenset[str]
    completed_players: frozenset[str]


def build_withdrawal_context(
    players: pd.DataFrame,
    money: pd.DataFrame,
) -> WithdrawalContext:
    """Build the one authoritative withdrawal and recipient population."""
    population = set(players["player_key"].dropna().astype(str))
    joined = money[
        money["player_key"].notna()
        & money["player_key"].astype(str).isin(population)
    ].copy()
    joined["player_key"] = joined["player_key"].astype(str)
    withdrawals = joined[joined["txn_type"].eq("WITHDRAWAL")].copy()
    completed = withdrawals[
        withdrawals["is_money_out"].fillna(False).astype(bool)
    ].copy()
    recipient = build_frame_linkage(
        completed,
        key_type="recipient_normalized",
        key_column="recipient_normalized",
        record_id_column="transaction_id",
    )
    return WithdrawalContext(
        withdrawals=withdrawals,
        completed=completed,
        recipient=recipient,
        withdrawal_players=frozenset(withdrawals["player_key"].unique()),
        completed_players=frozenset(completed["player_key"].unique()),
    )
