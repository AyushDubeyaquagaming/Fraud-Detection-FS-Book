"""Rule scoring and reviewer-facing flags for Phase 4.

The scorer is intentionally simple: it reads the frozen Phase 3 feature outputs,
evaluates config-driven rules, applies the within-group supporting-rule gate,
and writes an explainable flag table plus a markdown case report. Thresholds
and points are placeholders until production calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import config


GROUP_SCORE_COLUMNS = {
    "ma": "multi_accounting_score",
    "pay": "payment_fraud_score",
    "bet": "betting_anomaly_score",
}


@dataclass(frozen=True)
class Rule:
    """One config-defined scoring rule over one feature scalar."""

    rule_id: str
    group: str
    feature: str
    op: str
    threshold: float | None
    points: float
    reason_template: str


@dataclass(frozen=True)
class ScoringResult:
    """Scored flags plus output paths when files were written."""

    flags: pd.DataFrame
    output_paths: dict[str, Path]


def load_feature_outputs(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the Phase 3 feature table and evidence store from disk."""
    root = data_dir or config.DATA_DIR
    return (
        pd.read_parquet(root / "player_features.parquet"),
        pd.read_parquet(root / "player_features_evidence.parquet"),
    )


def build_rules(scoring_cfg: dict[str, Any]) -> list[Rule]:
    """Parse the `scoring.rules` config block into executable rules."""
    weights = scoring_cfg.get("weights", {})
    thresholds = scoring_cfg.get("rule_thresholds", {})
    rules: list[Rule] = []
    for raw in scoring_cfg.get("rules", []):
        rule_id = raw["rule_id"]
        threshold_key = raw.get("threshold_key")
        rules.append(
            Rule(
                rule_id=rule_id,
                group=_group_from_feature(raw["feature"]),
                feature=raw["feature"],
                op=raw["op"],
                threshold=thresholds.get(threshold_key) if threshold_key else None,
                points=float(weights.get(rule_id, 0)),
                reason_template=raw["reason_template"],
            )
        )
    return rules


def score_players(
    *,
    player_features: pd.DataFrame | None = None,
    feature_evidence: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    write_outputs: bool = True,
    cfg: dict[str, Any] | None = None,
) -> ScoringResult:
    """Score players from Phase 3 feature outputs and optionally write flags.

    Tests pass in synthetic feature/evidence frames directly. Production-like
    runs omit them, which reads `data/player_features.parquet` and
    `data/player_features_evidence.parquet`; no live data source is touched.
    """
    cfg = cfg or config.load_config()
    scoring_cfg = cfg["scoring"]
    if player_features is None or feature_evidence is None:
        loaded_features, loaded_evidence = load_feature_outputs(output_dir)
        player_features = loaded_features if player_features is None else player_features
        feature_evidence = loaded_evidence if feature_evidence is None else feature_evidence

    rules = build_rules(scoring_cfg)
    evidence_lookup = _build_evidence_lookup(feature_evidence)
    flags = _score_frame(player_features.copy(), rules, evidence_lookup, scoring_cfg)
    paths = write_scoring_outputs(flags, output_dir) if write_outputs else {}
    return ScoringResult(flags=flags, output_paths=paths)


def write_scoring_outputs(flags: pd.DataFrame, output_dir: Path | None = None) -> dict[str, Path]:
    """Write `flags.parquet` and the reviewer-oriented markdown report."""
    target = output_dir or config.DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "flags": target / "flags.parquet",
        "report": target / "flags_report.md",
    }
    flags.to_parquet(paths["flags"], index=False)
    write_flags_report(flags, paths["report"])
    return paths


def write_flags_report(
    flags: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int | None = None,
) -> None:
    """Write readable top-N case files for reviewer inspection."""
    cfg_top_n = config.load_config().get("scoring", {}).get("report_top_n", 20)
    limit = top_n if top_n is not None else int(cfg_top_n)
    ranked = flags.sort_values(["overall_risk", "player_key"], ascending=[False, True]).head(limit)

    lines = [
        "# Phase 4 Flags Report",
        "",
        "Thresholds, weights, and bands are PLACEHOLDER. Dev flags are a plumbing check only.",
        "",
    ]
    for row in ranked.itertuples(index=False):
        fired = json.loads(row.fired_rules)
        unmeasured = json.loads(row.unmeasured_features)
        lines.extend(
            [
                f"## {row.player_key}",
                "",
                f"- Band: {row.band}",
                f"- Overall risk: {row.overall_risk:g}",
                f"- Multi-accounting: {row.multi_accounting_score:g}",
                f"- Payment fraud: {row.payment_fraud_score:g}",
                f"- Betting anomaly: {row.betting_anomaly_score:g}",
                f"- IsolationForest score (untrusted, weight 0): {row.iforest_score_untrusted:g}",
                "",
                "### Fired Rules",
            ]
        )
        if fired:
            for item in fired:
                lines.append(f"- {item['rule_id']} ({item['points']:g}): {item['reason']}")
        else:
            lines.append("- None")
        lines.extend(["", "### Not Measurable"])
        if unmeasured:
            for item in unmeasured:
                lines.append(f"- {item['feature_name']}: {item['null_reason']}")
        else:
            lines.append("- None")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _score_frame(
    features: pd.DataFrame,
    rules: list[Rule],
    evidence_lookup: dict[tuple[str, str], list[dict[str, Any]]],
    scoring_cfg: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rules_by_group: dict[str, list[Rule]] = {"ma": [], "pay": [], "bet": []}
    for rule in rules:
        rules_by_group.setdefault(rule.group, []).append(rule)

    for feature_row in features.to_dict(orient="records"):
        player_key = str(feature_row["player_key"])
        fired_rules: list[dict[str, Any]] = []
        group_scores = {score_col: 0.0 for score_col in GROUP_SCORE_COLUMNS.values()}

        for group, group_rules in rules_by_group.items():
            scoring_rules = [
                rule for rule in group_rules if _feature_role(feature_row, rule.feature) == "scoring"
            ]
            supporting_rules = [
                rule for rule in group_rules if _feature_role(feature_row, rule.feature) == "supporting"
            ]
            group_fired = _evaluate_rules(player_key, feature_row, scoring_rules, evidence_lookup)
            if group_fired:
                group_fired.extend(
                    _evaluate_rules(player_key, feature_row, supporting_rules, evidence_lookup)
                )
            fired_rules.extend(group_fired)
            group_scores[GROUP_SCORE_COLUMNS[group]] = sum(item["points"] for item in group_fired)

        overall_risk = sum(group_scores.values())
        row = {
            "player_key": player_key,
            **group_scores,
            "overall_risk": overall_risk,
            "band": _risk_band(overall_risk, scoring_cfg.get("bands", {})),
            "iforest_score_untrusted": 0.0,
            "fired_rules": json.dumps(fired_rules, sort_keys=True, default=str),
            "unmeasured_features": json.dumps(_unmeasured_features(feature_row), sort_keys=True),
            "feature_snapshot": json.dumps(_feature_snapshot(feature_row), sort_keys=True, default=str),
        }
        rows.append(row)

    flags = pd.DataFrame(rows)
    flags["iforest_score_untrusted"] = _iforest_scores(features, scoring_cfg.get("iforest", {}))
    return flags.sort_values("player_key", kind="mergesort").reset_index(drop=True)


def _evaluate_rules(
    player_key: str,
    feature_row: dict[str, Any],
    rules: list[Rule],
    evidence_lookup: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    fired: list[dict[str, Any]] = []
    for rule in rules:
        value = feature_row.get(rule.feature)
        if _is_null(value) or not _condition_met(value, rule.op, rule.threshold):
            continue
        evidence = evidence_lookup.get((player_key, rule.feature), [])
        if not evidence:
            raise ValueError(f"Rule {rule.rule_id} fired without evidence.")
        reason = _format_reason(rule, value, evidence)
        if not reason:
            raise ValueError(f"Rule {rule.rule_id} fired without a reason.")
        fired.append(
            {
                "rule_id": rule.rule_id,
                "group": rule.group,
                "feature": rule.feature,
                "points": rule.points,
                "reason": reason,
                "feature_value": _json_scalar(value),
            }
        )
    return fired


def _format_reason(rule: Rule, value: Any, evidence: list[dict[str, Any]]) -> str:
    context = _reason_context(value, evidence)
    context.update({"points": rule.points, "threshold": rule.threshold})
    try:
        return rule.reason_template.format(**context)
    except KeyError as exc:
        raise ValueError(f"Missing reason placeholder {exc} for {rule.rule_id}.") from exc


def _reason_context(value: Any, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a forgiving formatting context from scalar value plus evidence."""
    scalar = _json_scalar(value)
    context: dict[str, Any] = {
        "value": scalar,
        "value_int": int(scalar) if isinstance(scalar, (int, float)) else scalar,
        "other_player_keys": "[]",
        "referred_player_keys": "[]",
        "shared_keys": "[]",
        "shared_key_types": "[]",
        "recipient_numbers": "[]",
        "deposit_ids": "[]",
        "withdrawal_ids": "[]",
        "bonus_staked_ticket_ids": "[]",
        "reason_text": "",
        "ticket_count": 0,
        "wins": 0,
        "settled_bets": 0,
        "min_stake": 0,
        "max_stake": 0,
    }
    if evidence:
        for key, raw_value in evidence[0].items():
            context[key] = _format_context_value(raw_value)
        context["other_player_keys"] = _list_value(
            _collect_values(evidence, "other_player_keys") + _collect_values(evidence, "other_player_key")
        )
        context["referred_player_keys"] = _list_value(
            _collect_values(evidence, "referred_player_keys") + _collect_values(evidence, "referred_player_key")
        )
        context["shared_keys"] = _list_value(_collect_values(evidence, "shared_key"))
        context["shared_key_types"] = _list_value(_collect_values(evidence, "shared_key_type"))
        context["recipient_numbers"] = _list_value(_collect_values(evidence, "recipient_number"))
        context["deposit_ids"] = _list_value(_collect_values(evidence, "deposit_id"))
        context["withdrawal_ids"] = _list_value(_collect_values(evidence, "withdrawal_id"))
        ticket_ids = _collect_values(evidence, "ticket_ids") or _collect_values(
            evidence, "bonus_staked_ticket_ids"
        )
        context["ticket_count"] = len(ticket_ids)
        if ticket_ids:
            context["bonus_staked_ticket_ids"] = _list_value(ticket_ids)
    return context


def _build_evidence_lookup(
    feature_evidence: pd.DataFrame,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in feature_evidence.itertuples(index=False):
        raw = row.feature_evidence
        if isinstance(raw, str):
            parsed = json.loads(raw)
        elif isinstance(raw, list):
            parsed = raw
        else:
            parsed = []
        lookup[(str(row.player_key), str(row.feature_name))] = parsed
    return lookup


def _condition_met(value: Any, op: str, threshold: float | None) -> bool:
    scalar = _json_scalar(value)
    if op == "is_true":
        return bool(scalar) is True
    if threshold is None:
        raise ValueError(f"Rule op {op} requires a threshold.")
    if op == "gt":
        return scalar > threshold
    if op == "gte":
        return scalar >= threshold
    if op == "lt":
        return scalar < threshold
    if op == "lte":
        return scalar <= threshold
    if op == "eq":
        return scalar == threshold
    raise ValueError(f"Unsupported rule op: {op}")


def _iforest_scores(features: pd.DataFrame, iforest_cfg: dict[str, Any]) -> list[float]:
    """Fit an untrusted IsolationForest plumbing score with weight zero."""
    if not iforest_cfg.get("enabled", True):
        return [0.0] * len(features)
    numeric = features[
        [
            col
            for col in features.columns
            if "__" not in col and col != "player_key" and pd.api.types.is_numeric_dtype(features[col])
        ]
    ].astype("float64")
    if numeric.empty or len(numeric) < 2:
        return [0.0] * len(features)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    try:
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(
            n_estimators=int(iforest_cfg.get("n_estimators", 100)),
            random_state=int(iforest_cfg.get("random_state", 42)),
        )
        return model.fit(numeric).score_samples(numeric).tolist()
    except Exception:
        # Scoring must remain deterministic even in a minimal environment. The
        # column still proves plumbing, and its configured weight is zero.
        return [0.0] * len(features)


def _unmeasured_features(feature_row: dict[str, Any]) -> list[dict[str, str]]:
    unmeasured: list[dict[str, str]] = []
    for column, value in feature_row.items():
        if not column.endswith("__null_reason") or _is_null(value):
            continue
        feature_name = column.removesuffix("__null_reason")
        unmeasured.append({"feature_name": feature_name, "null_reason": str(value)})
    return sorted(unmeasured, key=lambda item: item["feature_name"])


def _feature_snapshot(feature_row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _json_scalar(value)
        for key, value in feature_row.items()
        if key != "player_key" and "__" not in key
    }


def _feature_role(feature_row: dict[str, Any], feature_name: str) -> str:
    role = feature_row.get(f"{feature_name}__scoring_role")
    return "" if _is_null(role) else str(role)


def _group_from_feature(feature_name: str) -> str:
    if feature_name.startswith("ma_"):
        return "ma"
    if feature_name.startswith("pay_"):
        return "pay"
    if feature_name.startswith("bet_"):
        return "bet"
    raise ValueError(f"Cannot infer scoring group from feature: {feature_name}")


def _risk_band(score: float, bands: dict[str, Any]) -> str:
    medium_cut = float(bands.get("medium_cut", 40))
    high_cut = float(bands.get("high_cut", 80))
    if score >= high_cut:
        return "high"
    if score >= medium_cut:
        return "medium"
    return "low"


def _collect_values(evidence: list[dict[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    for item in evidence:
        if key not in item:
            continue
        raw = item[key]
        if isinstance(raw, list):
            values.extend(raw)
        elif raw is not None:
            values.append(raw)
    return sorted(dict.fromkeys(values), key=str)


def _format_context_value(value: Any) -> Any:
    if isinstance(value, list):
        return _list_value(value)
    return value


def _list_value(values: list[Any]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _json_scalar(value: Any) -> Any:
    if _is_null(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _is_null(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
