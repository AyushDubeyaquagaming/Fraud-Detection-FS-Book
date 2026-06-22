"""Reusable one-hop linkage indexes for relational player features."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class LinkageRow:
    """A player-to-shared-key observation with optional source record ID."""

    player_key: str
    shared_key: str
    record_id: str | None = None


@dataclass(frozen=True)
class LinkageIndex:
    """Direct shared-key index; never expands to connected components."""

    key_type: str
    key_to_players: dict[str, frozenset[str]]
    player_to_keys: dict[str, frozenset[str]]
    key_player_records: dict[str, dict[str, tuple[str, ...]]]

    def linked_players(self, player_key: str) -> frozenset[str]:
        """Return players sharing at least one of this player's own keys."""
        linked: set[str] = set()
        for shared_key in self.player_to_keys.get(player_key, frozenset()):
            linked.update(self.key_to_players.get(shared_key, frozenset()))
        linked.discard(player_key)
        return frozenset(linked)

    def shared_groups(self, player_key: str) -> list[dict[str, object]]:
        """Return reviewer evidence for this player's directly shared keys."""
        evidence: list[dict[str, object]] = []
        for shared_key in sorted(self.player_to_keys.get(player_key, frozenset())):
            players = self.key_to_players.get(shared_key, frozenset())
            others = sorted(players - {player_key})
            if not others:
                continue
            records = self.key_player_records.get(shared_key, {})
            evidence.append(
                {
                    "shared_key_type": self.key_type,
                    "shared_key": shared_key,
                    "other_player_keys": others,
                    "source_record_ids": list(records.get(player_key, ())),
                    "linked_source_record_ids": {
                        other: list(records.get(other, ())) for other in others
                    },
                }
            )
        return evidence

    def shared_keys_between(self, player_key: str, other_player_key: str) -> list[str]:
        """Return direct shared keys between two players."""
        own = self.player_to_keys.get(player_key, frozenset())
        other = self.player_to_keys.get(other_player_key, frozenset())
        return sorted(own & other)

    def with_max_cardinality(self, max_players: int) -> "LinkageIndex":
        """Return an index containing only keys observed on at most N players."""
        if max_players < 1:
            raise ValueError("max_players must be positive.")
        allowed = {
            key
            for key, players in self.key_to_players.items()
            if len(players) <= max_players
        }
        key_to_players = {
            key: players
            for key, players in self.key_to_players.items()
            if key in allowed
        }
        player_to_keys: dict[str, frozenset[str]] = {}
        for player, keys in self.player_to_keys.items():
            kept = frozenset(key for key in keys if key in allowed)
            if kept:
                player_to_keys[player] = kept
        return LinkageIndex(
            key_type=self.key_type,
            key_to_players=key_to_players,
            player_to_keys=player_to_keys,
            key_player_records={
                key: player_records
                for key, player_records in self.key_player_records.items()
                if key in allowed
            },
        )


def build_linkage_index(key_type: str, rows: Iterable[LinkageRow]) -> LinkageIndex:
    """Build a direct player/shared-key index from normalized observations."""
    key_players: dict[str, set[str]] = defaultdict(set)
    player_keys: dict[str, set[str]] = defaultdict(set)
    records: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for row in rows:
        if not row.player_key or not row.shared_key:
            continue
        key_players[row.shared_key].add(row.player_key)
        player_keys[row.player_key].add(row.shared_key)
        if row.record_id:
            records[row.shared_key][row.player_key].add(row.record_id)

    return LinkageIndex(
        key_type=key_type,
        key_to_players={key: frozenset(players) for key, players in key_players.items()},
        player_to_keys={player: frozenset(keys) for player, keys in player_keys.items()},
        key_player_records={
            key: {
                player: tuple(sorted(record_ids))
                for player, record_ids in player_records.items()
            }
            for key, player_records in records.items()
        },
    )


def build_frame_linkage(
    frame: pd.DataFrame,
    *,
    key_type: str,
    key_column: str,
    record_id_column: str | None = None,
) -> LinkageIndex:
    """Build a linkage index from a filtered player-keyed frame."""
    rows: list[LinkageRow] = []
    for row in frame.to_dict("records"):
        player_key = row.get("player_key")
        shared_key = row.get(key_column)
        if pd.isna(player_key) or pd.isna(shared_key):
            continue
        record_id = row.get(record_id_column) if record_id_column else None
        rows.append(
            LinkageRow(
                player_key=str(player_key),
                shared_key=str(shared_key),
                record_id=None if record_id is None or pd.isna(record_id) else str(record_id),
            )
        )
    return build_linkage_index(key_type, rows)
