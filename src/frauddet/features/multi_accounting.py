"""Multi-accounting feature group built from the frozen Phase 3 snapshot."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .. import config
from ..snapshot import load_snapshot
from .linkage import LinkageIndex, build_frame_linkage
from .schema import FeatureResult, FeatureScoringRole, FeatureStrength


_VALID_FINGERPRINT = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class FeatureSpec:
    strength: FeatureStrength
    scoring_role: FeatureScoringRole
    value_kind: str


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
    nin: LinkageIndex
    email: LinkageIndex
    device: LinkageIndex
    recipient: LinkageIndex
    phone: LinkageIndex
    completed_withdrawal_players: frozenset[str]

    def corroborating(self) -> dict[str, LinkageIndex]:
        """Linkages allowed to corroborate referral and co-creation signals."""
        return {
            "nin_hash": self.nin,
            "email_hash": self.email,
            "fingerprint": self.device,
            "recipient_normalized": self.recipient,
        }


@dataclass(frozen=True)
class FeatureBuildResult:
    player_features: pd.DataFrame
    feature_evidence: pd.DataFrame
    output_paths: dict[str, Path]


def build_multi_account_linkages(
    players: pd.DataFrame,
    logins: pd.DataFrame,
    money: pd.DataFrame,
) -> MultiAccountLinkages:
    """Build reusable population-wide one-hop linkage indexes."""
    population = set(players["player_key"].astype(str))
    player_rows = players[players["player_key"].astype(str).isin(population)]

    valid_logins = logins[
        logins["player_key"].notna()
        & logins["player_key"].astype(str).isin(population)
        & logins["user_type"].eq("PLAYER")
        & logins["fingerprint"].notna()
        & logins["fingerprint"].astype(str).str.fullmatch(_VALID_FINGERPRINT)
    ]
    completed_withdrawals = money[
        money["player_key"].notna()
        & money["player_key"].astype(str).isin(population)
        & money["txn_type"].eq("WITHDRAWAL")
        & money["is_money_out"].fillna(False).astype(bool)
    ]

    return MultiAccountLinkages(
        nin=build_frame_linkage(
            player_rows, key_type="nin_hash", key_column="nin_hash"
        ),
        email=build_frame_linkage(
            player_rows, key_type="email_hash", key_column="email_hash"
        ),
        device=build_frame_linkage(
            valid_logins,
            key_type="fingerprint",
            key_column="fingerprint",
            record_id_column="source_id",
        ),
        recipient=build_frame_linkage(
            completed_withdrawals,
            key_type="recipient_normalized",
            key_column="recipient_normalized",
            record_id_column="transaction_id",
        ),
        phone=build_frame_linkage(
            player_rows, key_type="normalized_phone", key_column="phone"
        ),
        completed_withdrawal_players=frozenset(
            completed_withdrawals["player_key"].astype(str).unique()
        ),
    )


def build_multi_accounting_features(
    *,
    players: pd.DataFrame | None = None,
    logins: pd.DataFrame | None = None,
    money: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    write_outputs: bool = True,
) -> FeatureBuildResult:
    """Build exactly the Phase 3 v1 multi-accounting feature group."""
    players = load_snapshot("players") if players is None else players.copy()
    logins = load_snapshot("logins") if logins is None else logins.copy()
    money = load_snapshot("money") if money is None else money.copy()

    players = players[players["player_key"].notna()].copy()
    players["player_key"] = players["player_key"].astype(str)
    if players["player_key"].duplicated().any():
        raise ValueError("players input must contain one row per player_key.")

    linkages = build_multi_account_linkages(players, logins, money)
    referred_players = _referred_players(players)
    created_at = players.set_index("player_key")["created_at"].to_dict()
    cocreation_minutes = float(
        config.load_config()["thresholds"]["ma_cocreation_window_minutes"]
    )

    feature_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, str]] = []
    for player_key in sorted(players["player_key"]):
        player = players.loc[players["player_key"].eq(player_key)].iloc[0]
        results = _player_results(
            player_key,
            player,
            linkages,
            referred_players,
            created_at,
            cocreation_minutes,
        )
        feature_row: dict[str, Any] = {"player_key": player_key}
        for feature_name in FEATURE_SPECS:
            result = results[feature_name]
            feature_row[feature_name] = result.feature_value
            feature_row[f"{feature_name}__null_reason"] = result.feature_null_reason
            feature_row[f"{feature_name}__strength"] = result.feature_strength
            feature_row[f"{feature_name}__scoring_role"] = result.feature_scoring_role
            evidence_rows.append(
                {
                    "player_key": player_key,
                    "feature_name": feature_name,
                    "feature_evidence": json.dumps(
                        result.feature_evidence,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ),
                }
            )
        feature_rows.append(feature_row)

    feature_frame = pd.DataFrame(feature_rows)
    _apply_feature_dtypes(feature_frame)
    evidence_frame = pd.DataFrame(
        evidence_rows,
        columns=["player_key", "feature_name", "feature_evidence"],
    ).sort_values(["player_key", "feature_name"], kind="mergesort").reset_index(drop=True)

    output_paths: dict[str, Path] = {}
    if write_outputs:
        target = output_dir or config.DATA_DIR
        target.mkdir(parents=True, exist_ok=True)
        output_paths = {
            "player_features": target / "player_features.parquet",
            "feature_evidence": target / "player_features_evidence.parquet",
        }
        feature_frame.to_parquet(output_paths["player_features"], index=False)
        evidence_frame.to_parquet(output_paths["feature_evidence"], index=False)

    return FeatureBuildResult(feature_frame, evidence_frame, output_paths)


def _player_results(
    player_key: str,
    player: pd.Series,
    linkages: MultiAccountLinkages,
    referred_players: dict[str, list[str]],
    created_at: dict[str, Any],
    cocreation_minutes: float,
) -> dict[str, FeatureResult]:
    results = {
        "ma_nin_shared_account_count": _link_count(linkages.nin, player_key, "strong", "scoring"),
        "ma_email_shared_account_count": _link_count(linkages.email, player_key, "strong", "scoring"),
        "ma_device_shared_account_count": _link_count(
            linkages.device,
            player_key,
            "strong",
            "scoring",
            null_reason="no_valid_fingerprint_logins",
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
        "ma_device_count": _device_count_result(player_key, linkages.device),
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
) -> FeatureResult:
    has_keys = bool(index.player_to_keys.get(player_key))
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


def _apply_feature_dtypes(frame: pd.DataFrame) -> None:
    for feature_name, spec in FEATURE_SPECS.items():
        if spec.value_kind == "count":
            frame[feature_name] = frame[feature_name].astype("Int64")
        elif spec.value_kind == "bool":
            frame[feature_name] = frame[feature_name].astype("boolean")
        frame[f"{feature_name}__null_reason"] = frame[
            f"{feature_name}__null_reason"
        ].astype("string")
        frame[f"{feature_name}__strength"] = frame[
            f"{feature_name}__strength"
        ].astype("string")
        frame[f"{feature_name}__scoring_role"] = frame[
            f"{feature_name}__scoring_role"
        ].astype("string")
