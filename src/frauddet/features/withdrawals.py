"""Shared completed-withdrawal population and recipient linkage.

Multi-accounting and payment features both need the same view of withdrawals.
This module builds that view once so "completed withdrawal" and "recipient
sharing" mean the same thing everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .linkage import LinkageIndex, build_frame_linkage


@dataclass(frozen=True)
class WithdrawalContext:
    """Pre-filtered withdrawal tables and recipient linkage indexes."""

    withdrawals: pd.DataFrame
    completed: pd.DataFrame
    recipient: LinkageIndex
    withdrawal_players: frozenset[str]
    completed_players: frozenset[str]


def build_withdrawal_context(
    players: pd.DataFrame,
    money: pd.DataFrame,
) -> WithdrawalContext:
    """Build the one authoritative withdrawal and recipient population.

    Only rows with a canonical `player_key` enter feature work. Completed
    withdrawals are identified by the Phase 2 `is_money_out` flag rather than
    rechecking raw statuses in every feature group.
    """
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
    # Recipient linkage powers both ma_withdrawal_recipient_shared_count and
    # the payment third-party-recipient features.
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
