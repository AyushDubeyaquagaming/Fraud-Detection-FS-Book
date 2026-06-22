"""Shared materialization and output helpers for feature groups."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .. import config
from .schema import FeatureResult, FeatureScoringRole, FeatureStrength


@dataclass(frozen=True)
class FeatureSpec:
    strength: FeatureStrength
    scoring_role: FeatureScoringRole
    value_kind: str


@dataclass(frozen=True)
class FeatureBuildResult:
    player_features: pd.DataFrame
    feature_evidence: pd.DataFrame
    output_paths: dict[str, Path]


def materialize_feature_group(
    player_keys: list[str],
    specs: dict[str, FeatureSpec],
    result_factory: Callable[[str], dict[str, FeatureResult]],
    *,
    output_dir: Path | None = None,
    write_outputs: bool = True,
) -> FeatureBuildResult:
    """Materialize one feature group using the standard wide/long contract."""
    feature_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, str]] = []
    for player_key in sorted(player_keys):
        results = result_factory(player_key)
        if set(results) != set(specs):
            raise ValueError("Feature result names do not match the declared specs.")
        feature_row: dict[str, object] = {"player_key": player_key}
        for feature_name, spec in specs.items():
            result = results[feature_name]
            if (
                result.feature_strength != spec.strength
                or result.feature_scoring_role != spec.scoring_role
            ):
                raise ValueError(f"Metadata mismatch for {feature_name}.")
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
    apply_feature_dtypes(feature_frame, specs)
    evidence_frame = pd.DataFrame(
        evidence_rows,
        columns=["player_key", "feature_name", "feature_evidence"],
    ).sort_values(["player_key", "feature_name"], kind="mergesort").reset_index(drop=True)
    paths = write_feature_outputs(feature_frame, evidence_frame, output_dir) if write_outputs else {}
    return FeatureBuildResult(feature_frame, evidence_frame, paths)


def combine_feature_build_results(
    *results: FeatureBuildResult,
    output_dir: Path | None = None,
    write_outputs: bool = True,
) -> FeatureBuildResult:
    """Combine independently built groups into the standard Phase 3 outputs."""
    if not results:
        raise ValueError("At least one feature result is required.")
    combined = results[0].player_features.copy()
    evidence_frames = [results[0].feature_evidence]
    for result in results[1:]:
        combined = combined.merge(
            result.player_features,
            on="player_key",
            how="inner",
            validate="one_to_one",
        )
        evidence_frames.append(result.feature_evidence)
    expected = set(results[0].player_features["player_key"])
    if set(combined["player_key"]) != expected or any(
        set(result.player_features["player_key"]) != expected for result in results
    ):
        raise ValueError("Feature groups must cover the same player population.")
    evidence = pd.concat(evidence_frames, ignore_index=True).sort_values(
        ["player_key", "feature_name"], kind="mergesort"
    ).reset_index(drop=True)
    if evidence.duplicated(["player_key", "feature_name"]).any():
        raise ValueError("Feature groups contain duplicate feature names.")
    paths = write_feature_outputs(combined, evidence, output_dir) if write_outputs else {}
    return FeatureBuildResult(combined, evidence, paths)


def apply_feature_dtypes(frame: pd.DataFrame, specs: dict[str, FeatureSpec]) -> None:
    for feature_name, spec in specs.items():
        if spec.value_kind == "count":
            frame[feature_name] = frame[feature_name].astype("Int64")
        elif spec.value_kind == "bool":
            frame[feature_name] = frame[feature_name].astype("boolean")
        elif spec.value_kind == "float":
            frame[feature_name] = frame[feature_name].astype("Float64")
        frame[f"{feature_name}__null_reason"] = frame[
            f"{feature_name}__null_reason"
        ].astype("string")
        frame[f"{feature_name}__strength"] = frame[
            f"{feature_name}__strength"
        ].astype("string")
        frame[f"{feature_name}__scoring_role"] = frame[
            f"{feature_name}__scoring_role"
        ].astype("string")


def write_feature_outputs(
    player_features: pd.DataFrame,
    feature_evidence: pd.DataFrame,
    output_dir: Path | None,
) -> dict[str, Path]:
    target = output_dir or config.DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "player_features": target / "player_features.parquet",
        "feature_evidence": target / "player_features_evidence.parquet",
    }
    player_features.to_parquet(paths["player_features"], index=False)
    feature_evidence.to_parquet(paths["feature_evidence"], index=False)
    return paths
