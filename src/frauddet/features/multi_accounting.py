"""Multi-accounting feature group built from the frozen Phase 3 snapshot.

The group looks for direct shared identifiers between player accounts: identity
document hashes, email hashes, device fingerprints, withdrawal recipients, and
phone collisions. Evidence always lists the directly linked player keys; there
is no transitive clustering.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .. import config
from ..snapshot import load_snapshot
from .linkage import LinkageIndex, build_frame_linkage
from .output import FeatureBuildResult, FeatureSpec, materialize_feature_group
from .schema import FeatureResult, FeatureScoringRole, FeatureStrength
from .withdrawals import WithdrawalContext, build_withdrawal_context


_VALID_FINGERPRINT = re.compile(r"^[0-9a-fA-F]{64}$")


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "ma_nin_shared_account_count": FeatureSpec("strong", "scoring", "count"),
    "ma_email_shared_account_count": FeatureSpec("strong", "scoring", "count"),
    "ma_device_shared_account_count": FeatureSpec("strong", "scoring", "count"),
    "ma_withdrawal_recipient_shared_count": FeatureSpec("strong", "scoring", "count"),
    "ma_identity_phone_collision_count": FeatureSpec("strong", "scoring", "count"),
    "ma_referred_by_linked_account": FeatureSpec("strong", "supporting", "bool"),
    "ma_referral_fanout_count": FeatureSpec("weak", "supporting", "count"),
    "ma_cocreated_linked_count": FeatureSpec("moderate", "supporting", "count"),
    "ma_device_count": FeatureSpec("weak", "context_only", "count"),
}


@dataclass(frozen=True)
class MultiAccountLinkages:
    """All population-wide linkage indexes needed by ma_ features."""

    nin: LinkageIndex
    email: LinkageIndex
    device_all: LinkageIndex
    device: LinkageIndex
    recipient: LinkageIndex
    phone: LinkageIndex
    completed_withdrawal_players: frozenset[str]

    def corroborating(self) -> dict[str, LinkageIndex]:
        """Linkages allowed to corroborate referral and co-creation signals.

        Device uses the capped index here. Very high-cardinality fingerprints
        are dev/office artifacts and should not turn referral/co-creation into
        near-universal hits.
        """
        return {
            "nin_hash": self.nin,
            "email_hash": self.email,
            "fingerprint": self.device,
            "recipient_normalized": self.recipient,
        }


def build_multi_account_linkages(
    players: pd.DataFrame,
    logins: pd.DataFrame,
    money: pd.DataFrame,
    *,
    device_max_cardinality: int | None = None,
    withdrawal_context: WithdrawalContext | None = None,
) -> MultiAccountLinkages:
    """Build reusable population-wide one-hop linkage indexes."""
    population = set(players["player_key"].astype(str))
    player_rows = players[players["player_key"].astype(str).isin(population)]

    # The flatten layer already converts the known "not-found" sentinel to null.
    # This extra format check keeps synthetic or malformed hashes out of linkage.
    valid_logins = logins[
        logins["player_key"].notna()
        & logins["player_key"].astype(str).isin(population)
        & logins["user_type"].eq("PLAYER")
        & logins["fingerprint"].notna()
        & logins["fingerprint"].astype(str).str.fullmatch(_VALID_FINGERPRINT)
    ]
    withdrawals = withdrawal_context or build_withdrawal_context(players, money)
    device_all = build_frame_linkage(
        valid_logins,
        key_type="fingerprint",
        key_column="fingerprint",
        record_id_column="source_id",
    )
    if device_max_cardinality is None:
        device_max_cardinality = int(
            config.load_config()["thresholds"]["ma_device_max_cardinality"]
        )

    # Keep the raw device index for device_count, but cap the linkage index used
    # as evidence. This preserves context without treating office devices as
    # proof of shared account control.
    return MultiAccountLinkages(
        nin=build_frame_linkage(
            player_rows, key_type="nin_hash", key_column="nin_hash"
        ),
        email=build_frame_linkage(
            player_rows, key_type="email_hash", key_column="email_hash"
        ),
        device_all=device_all,
        device=device_all.with_max_cardinality(device_max_cardinality),
        recipient=withdrawals.recipient,
        phone=build_frame_linkage(
            player_rows, key_type="normalized_phone", key_column="phone"
        ),
        completed_withdrawal_players=withdrawals.completed_players,
    )


def build_multi_accounting_features(
    *,
    players: pd.DataFrame | None = None,
    logins: pd.DataFrame | None = None,
    money: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    write_outputs: bool = True,
    device_max_cardinality: int | None = None,
    withdrawal_context: WithdrawalContext | None = None,
) -> FeatureBuildResult:
    """Build exactly the Phase 3 v1 multi-accounting feature group.

    Players without a canonical key are excluded before feature generation.
    Missing NIN/email means "no shared identifier observed" and returns zero;
    missing device fingerprints are null because the device signal was not
    measurable for that player.
    """
    players = load_snapshot("players") if players is None else players.copy()
    logins = load_snapshot("logins") if logins is None else logins.copy()
    money = load_snapshot("money") if money is None else money.copy()

    players = players[players["player_key"].notna()].copy()
    players["player_key"] = players["player_key"].astype(str)
    if players["player_key"].duplicated().any():
        raise ValueError("players input must contain one row per player_key.")

    linkages = build_multi_account_linkages(
        players,
        logins,
        money,
        device_max_cardinality=device_max_cardinality,
        withdrawal_context=withdrawal_context,
    )
    referred_players = _referred_players(players)
    created_at = players.set_index("player_key")["created_at"].to_dict()
    cocreation_minutes = float(
        config.load_config()["thresholds"]["ma_cocreation_window_minutes"]
    )

    players_by_key = players.set_index("player_key", drop=False)

    def results_for(player_key: str) -> dict[str, FeatureResult]:
        return _player_results(
            player_key,
            players_by_key.loc[player_key],
            linkages,
            referred_players,
            created_at,
            cocreation_minutes,
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
    player: pd.Series,
    linkages: MultiAccountLinkages,
    referred_players: dict[str, list[str]],
    created_at: dict[str, Any],
    cocreation_minutes: float,
) -> dict[str, FeatureResult]:
    """Compute all ma_ feature results for one player."""
    results = {
        "ma_nin_shared_account_count": _link_count(linkages.nin, player_key, "strong", "scoring"),
        "ma_email_shared_account_count": _link_count(linkages.email, player_key, "strong", "scoring"),
        "ma_device_shared_account_count": _link_count(
            linkages.device,
            player_key,
            "strong",
            "scoring",
            null_reason="no_valid_fingerprint_logins",
            measurement_index=linkages.device_all,
        ),
        "ma_withdrawal_recipient_shared_count": _link_count(
            linkages.recipient,
            player_key,
            "strong",
            "scoring",
            null_reason=(
                None
                if player_key in linkages.completed_withdrawal_players
                else "no_completed_withdrawals"
            ),
            null_when_no_keys=False,
        ),
        "ma_identity_phone_collision_count": _link_count(
            linkages.phone, player_key, "strong", "scoring"
        ),
        "ma_referred_by_linked_account": _referred_by_linked(
            player_key, player.get("referred_by_key"), linkages
        ),
        "ma_referral_fanout_count": _fanout_result(
            referred_players.get(player_key, [])
        ),
        "ma_cocreated_linked_count": _cocreated_result(
            player_key,
            created_at,
            linkages,
            cocreation_minutes,
        ),
        "ma_device_count": _device_count_result(player_key, linkages.device_all),
    }
    return results


def _link_count(
    index: LinkageIndex,
    player_key: str,
    strength: FeatureStrength,
    scoring_role: FeatureScoringRole,
    *,
    null_reason: str | None = None,
    null_when_no_keys: bool = True,
    measurement_index: LinkageIndex | None = None,
) -> FeatureResult:
    """Count directly linked players for one shared-key index."""
    measured_by = measurement_index or index
    has_keys = bool(measured_by.player_to_keys.get(player_key))
    if null_reason and (not has_keys if null_when_no_keys else True):
        return FeatureResult(None, [], null_reason, strength, scoring_role)
    linked = index.linked_players(player_key)
    return FeatureResult(
        len(linked),
        index.shared_groups(player_key),
        None,
        strength,
        scoring_role,
    )


def _referred_players(players: pd.DataFrame) -> dict[str, list[str]]:
    """Return referrer -> directly referred player keys."""
    children: dict[str, list[str]] = defaultdict(list)
    population = set(players["player_key"])
    for row in players[["player_key", "referred_by_key"]].to_dict("records"):
        referrer = row.get("referred_by_key")
        if pd.isna(referrer):
            continue
        referrer = str(referrer)
        if referrer in population:
            children[referrer].append(str(row["player_key"]))
    return {key: sorted(set(values)) for key, values in children.items()}


def _direct_linkage_evidence(
    player_key: str,
    other_player_key: str,
    linkages: MultiAccountLinkages,
) -> list[dict[str, str]]:
    """Show which direct keys connect two specific players."""
    evidence: list[dict[str, str]] = []
    for key_type, index in linkages.corroborating().items():
        for shared_key in index.shared_keys_between(player_key, other_player_key):
            evidence.append({"shared_key_type": key_type, "shared_key": shared_key})
    return evidence


def _referred_by_linked(
    player_key: str,
    referred_by_key: Any,
    linkages: MultiAccountLinkages,
) -> FeatureResult:
    """Flag referral edges that are backed by another shared identifier."""
    if referred_by_key is None or pd.isna(referred_by_key):
        return FeatureResult(False, [], None, "strong", "supporting")
    referrer = str(referred_by_key)
    corroborating = _direct_linkage_evidence(player_key, referrer, linkages)
    evidence = (
        [{"referrer_player_key": referrer, "corroborating_linkages": corroborating}]
        if corroborating
        else []
    )
    return FeatureResult(bool(corroborating), evidence, None, "strong", "supporting")


def _fanout_result(referred_player_keys: list[str]) -> FeatureResult:
    """Count direct referrals made by this player."""
    evidence = (
        [{"referred_player_keys": referred_player_keys}]
        if referred_player_keys
        else []
    )
    return FeatureResult(
        len(referred_player_keys), evidence, None, "weak", "supporting"
    )


def _cocreated_result(
    player_key: str,
    created_at: dict[str, Any],
    linkages: MultiAccountLinkages,
    window_minutes: float,
) -> FeatureResult:
    """Count directly linked accounts created close together in time."""
    own_created = pd.to_datetime(created_at.get(player_key), utc=True, errors="coerce")
    candidates: set[str] = set()
    for index in linkages.corroborating().values():
        candidates.update(index.linked_players(player_key))

    evidence: list[dict[str, Any]] = []
    if pd.notna(own_created):
        for other in sorted(candidates):
            other_created = pd.to_datetime(created_at.get(other), utc=True, errors="coerce")
            if pd.isna(other_created):
                continue
            gap_minutes = abs((other_created - own_created).total_seconds()) / 60
            if gap_minutes <= window_minutes:
                evidence.append(
                    {
                        "other_player_key": other,
                        "player_created_at": own_created.isoformat(),
                        "other_created_at": other_created.isoformat(),
                        "gap_minutes": gap_minutes,
                        "shared_linkages": _direct_linkage_evidence(
                            player_key, other, linkages
                        ),
                    }
                )
    return FeatureResult(len(evidence), evidence, None, "moderate", "supporting")


def _device_count_result(player_key: str, device: LinkageIndex) -> FeatureResult:
    """Count distinct valid fingerprints seen for this player."""
    fingerprints = sorted(device.player_to_keys.get(player_key, frozenset()))
    if not fingerprints:
        return FeatureResult(
            None,
            [],
            "no_valid_fingerprint_logins",
            "weak",
            "context_only",
        )
    return FeatureResult(
        len(fingerprints),
        [{"fingerprints": fingerprints}],
        None,
        "weak",
        "context_only",
    )
