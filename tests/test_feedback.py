"""Tests for the Phase 5 local feedback loop."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.feedback import (
    DECISIONS,
    CASE_COLUMNS,
    REVIEW_COLUMNS,
    LocalFeedbackStore,
    build_human_discovered_case,
    build_labels,
    build_model_flagged_cases,
    case_id_for,
    get_routing_bands,
    get_ruleset_version,
    get_snapshot_id,
    record_review,
    run_simulated_feedback_loop,
)


def _flags():
    """Small Phase 4-like flag table with routed and non-routed rows."""
    return pd.DataFrame(
        [
            _flag_row("p_low_1", "low", 0, []),
            _flag_row("p_medium", "medium", 40, [{"rule_id": "R-MA-03"}]),
            _flag_row("p_high", "high", 85, [{"rule_id": "R-PAY-01"}]),
            _flag_row("p_low_2", "low", 10, [{"rule_id": "R-MA-07"}]),
        ]
    )


def _flag_row(player_key, band, overall_risk, fired_rules):
    return {
        "player_key": player_key,
        "multi_accounting_score": overall_risk if band != "high" else 45,
        "payment_fraud_score": 0 if band != "high" else 40,
        "betting_anomaly_score": 0,
        "overall_risk": overall_risk,
        "band": band,
        "iforest_score_untrusted": 0.0,
        "fired_rules": json.dumps(fired_rules),
        "unmeasured_features": "[]",
        "feature_snapshot": "{}",
    }


def _cfg(tmp_path):
    return {
        "phase3": {"input_dir": "data/snapshot_phase3_2026-06-19b"},
        "scoring": {"ruleset_version": "test_rules_v1"},
        "feedback": {
            "output_dir": str(tmp_path / "feedback"),
            "sqlite_path": str(tmp_path / "feedback" / "feedback.sqlite"),
            "routing_bands": ["medium", "high"],
        },
    }


def test_routing_gate_creates_cases_only_for_configured_bands(tmp_path):
    cfg = _cfg(tmp_path)
    cases = build_model_flagged_cases(
        _flags(),
        run_id="run_a",
        ruleset_version=get_ruleset_version(cfg),
        snapshot_id=get_snapshot_id(cfg),
        routing_bands=get_routing_bands(cfg),
        created_at="2026-06-25T00:00:00Z",
    )

    assert len(cases) == 2
    assert set(cases["player_key"]) == {"p_medium", "p_high"}
    assert set(cases["detection_source"]) == {"model_flagged"}
    assert cases["model_output_available"].all()
    assert cases["model_routed"].all()


def test_case_id_is_unique_per_flagging_event():
    same_run = case_id_for("run_a", "player_1", "model_flagged")
    repeated = case_id_for("run_a", "player_1", "model_flagged")
    new_run = case_id_for("run_b", "player_1", "model_flagged")
    human_channel = case_id_for("run_a", "player_1", "human_discovered")

    assert same_run == repeated
    assert len({same_run, new_run, human_channel}) == 3


def test_local_store_append_only_reviews_and_status_coupling(tmp_path):
    store = LocalFeedbackStore(tmp_path / "feedback.sqlite")
    case = build_model_flagged_cases(
        _flags().iloc[[1]],
        run_id="run_a",
        ruleset_version="test_rules_v1",
        snapshot_id="snapshot_phase3_2026-06-19b",
        routing_bands={"medium", "high"},
        created_at="2026-06-25T00:00:00Z",
    )
    store.write_cases(case)
    case_id = str(case.iloc[0]["case_id"])

    record_review(
        store,
        case_id=case_id,
        decision="needs_more_review",
        reviewed_at="2026-06-25T00:01:00Z",
    )
    assert store.read_cases().set_index("case_id").loc[case_id, "status"] == "in_review"

    record_review(
        store,
        case_id=case_id,
        decision="true_positive",
        reviewed_at="2026-06-25T00:02:00Z",
        fraud_type="multi_accounting",
    )

    assert len(store.read_reviews()) == 2
    assert store.read_cases().set_index("case_id").loc[case_id, "status"] == "closed"


def test_labels_latest_review_wins_and_untrainable_reviews_are_retained(tmp_path):
    store = LocalFeedbackStore(tmp_path / "feedback.sqlite")
    cases = build_model_flagged_cases(
        _flags().iloc[[1, 2]],
        run_id="run_a",
        ruleset_version="test_rules_v1",
        snapshot_id="snapshot_phase3_2026-06-19b",
        routing_bands={"medium", "high"},
        created_at="2026-06-25T00:00:00Z",
    )
    store.write_cases(cases)
    first_case = str(cases.iloc[0]["case_id"])
    second_case = str(cases.iloc[1]["case_id"])

    record_review(store, case_id=first_case, decision="needs_more_review", reviewed_at="2026-06-25T00:01:00Z")
    record_review(store, case_id=first_case, decision="false_positive", reviewed_at="2026-06-25T00:02:00Z")
    record_review(
        store,
        case_id=second_case,
        decision="insufficient_evidence",
        reviewed_at="2026-06-25T00:03:00Z",
    )

    all_reviewed, trainable = build_labels(store)

    assert len(store.read_reviews()) == 3
    assert len(all_reviewed) == 2
    assert all_reviewed.set_index("case_id").loc[first_case, "decision"] == "false_positive"
    assert set(trainable["case_id"]) == {first_case}
    assert second_case in set(store.read_cases()["case_id"])


def test_two_intake_channels_and_false_negative_is_emergent(tmp_path):
    store = LocalFeedbackStore(tmp_path / "feedback.sqlite")
    flags = _flags()
    model_cases = build_model_flagged_cases(
        flags,
        run_id="run_a",
        ruleset_version="test_rules_v1",
        snapshot_id="snapshot_phase3_2026-06-19b",
        routing_bands={"medium", "high"},
        created_at="2026-06-25T00:00:00Z",
    )
    human_case = build_human_discovered_case(
        "p_low_1",
        flags,
        run_id="run_a",
        ruleset_version="test_rules_v1",
        snapshot_id="snapshot_phase3_2026-06-19b",
        routing_bands={"medium", "high"},
        created_at="2026-06-25T00:00:00Z",
    )
    store.write_cases(pd.concat([model_cases, pd.DataFrame([human_case])], ignore_index=True))

    record_review(
        store,
        case_id=human_case["case_id"],
        decision="true_positive",
        reviewed_at="2026-06-25T00:01:00Z",
        fraud_type="multi_accounting",
    )
    _, trainable = build_labels(store)
    row = trainable.iloc[0]

    assert "false_negative" not in DECISIONS
    assert row["detection_source"] == "human_discovered"
    assert bool(row["model_output_available"]) is True
    assert bool(row["model_routed"]) is False
    assert bool(row["is_false_negative"]) is True


def test_human_discovered_case_without_model_output_is_valid():
    case = build_human_discovered_case(
        "not_in_flags",
        _flags(),
        run_id="run_a",
        ruleset_version="test_rules_v1",
        snapshot_id="snapshot_phase3_2026-06-19b",
        routing_bands={"medium", "high"},
        created_at="2026-06-25T00:00:00Z",
    )

    assert case["detection_source"] == "human_discovered"
    assert case["model_output_available"] is False
    assert case["model_routed"] is False
    assert case["overall_risk"] is None


class FakeStore:
    """Protocol-shaped store used to keep loop helpers independent of SQLite."""

    def __init__(self, cases):
        self.cases = cases.copy()
        self.reviews = pd.DataFrame(columns=REVIEW_COLUMNS)

    def write_cases(self, cases):
        self.cases = pd.concat([self.cases, cases], ignore_index=True)

    def read_cases(self):
        return self.cases

    def write_review(self, review):
        self.reviews = pd.concat([self.reviews, pd.DataFrame([review])], ignore_index=True)

    def update_case_status(self, case_id, status):
        self.cases.loc[self.cases["case_id"] == case_id, "status"] = status

    def read_reviews(self):
        return self.reviews


def test_loop_helpers_use_feedback_store_interface():
    case = pd.DataFrame(
        [
            {
                **{column: None for column in CASE_COLUMNS},
                "case_id": "case_fake",
                "run_id": "run_a",
                "player_key": "p1",
                "created_at": "2026-06-25T00:00:00Z",
                "fired_rules": "[]",
                "unmeasured_features": "[]",
                "feature_snapshot": "{}",
                "status": "open",
                "ruleset_version": "test_rules_v1",
                "snapshot_id": "snapshot_phase3_2026-06-19b",
                "detection_source": "model_flagged",
                "model_output_available": True,
                "model_routed": True,
            }
        ]
    )
    store = FakeStore(case)

    record_review(
        store,
        case_id="case_fake",
        decision="false_positive",
        reviewed_at="2026-06-25T00:01:00Z",
    )
    _, trainable = build_labels(store)

    assert trainable.iloc[0]["case_id"] == "case_fake"
    assert store.cases.iloc[0]["status"] == "closed"


def test_simulated_loop_writes_run_addressed_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    flags_path = tmp_path / "flags.parquet"
    _flags().to_parquet(flags_path, index=False)

    run_simulated_feedback_loop(
        flags_path=flags_path,
        output_dir=tmp_path / "feedback",
        sqlite_path=tmp_path / "feedback" / "feedback.sqlite",
        cfg=cfg,
        run_id="prior_feedback_run",
    )
    result = run_simulated_feedback_loop(
        flags_path=flags_path,
        output_dir=tmp_path / "feedback",
        sqlite_path=tmp_path / "feedback" / "feedback.sqlite",
        cfg=cfg,
        run_id="test_feedback_run",
    )

    assert result.output_paths["sqlite"].exists()
    assert result.output_paths["labels"].exists()
    assert result.output_paths["run_manifest"].exists()
    assert result.run_manifest["flags_row_count"] == 4
    assert result.run_manifest["routed_case_count"] == 2
    assert result.run_manifest["case_row_count"] == 3
    assert result.run_manifest["review_row_count"] == 3
    assert len(LocalFeedbackStore(tmp_path / "feedback" / "feedback.sqlite").read_cases()) == 6
    assert result.run_manifest["case_row_count"] < result.run_manifest["flags_row_count"]
    assert result.run_manifest["false_negative_label_count"] == 1
    assert "precision" not in result.run_manifest
    assert "recall" not in result.run_manifest
    assert {"ruleset_version", "snapshot_id", "detection_source", "model_routed"} <= set(
        result.labels.columns
    )


def test_review_validation_rejects_boolean_or_false_negative_decision(tmp_path):
    store = LocalFeedbackStore(tmp_path / "feedback.sqlite")
    case = build_model_flagged_cases(
        _flags().iloc[[1]],
        run_id="run_a",
        ruleset_version="test_rules_v1",
        snapshot_id="snapshot_phase3_2026-06-19b",
        routing_bands={"medium", "high"},
        created_at="2026-06-25T00:00:00Z",
    )
    store.write_cases(case)

    with pytest.raises(ValueError, match="Unsupported review decision"):
        record_review(
            store,
            case_id=str(case.iloc[0]["case_id"]),
            decision="false_negative",
            reviewed_at="2026-06-25T00:01:00Z",
        )
