"""Local-first feedback loop for Phase 5.

The feedback loop turns routed flags into review cases, stores append-only
human decisions, and writes a versioned label artifact. The core functions use
the FeedbackStore protocol only, so a future Mongo store can replace the local
SQLite store without rewriting case/review/label logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Protocol

import pandas as pd

from . import config


CASE_COLUMNS = [
    "case_id",
    "run_id",
    "player_key",
    "created_at",
    "overall_risk",
    "band",
    "multi_accounting_score",
    "payment_fraud_score",
    "betting_anomaly_score",
    "fired_rules",
    "unmeasured_features",
    "feature_snapshot",
    "status",
    "ruleset_version",
    "snapshot_id",
    "detection_source",
    "model_output_available",
    "model_routed",
]
REVIEW_COLUMNS = [
    "review_id",
    "case_id",
    "decision",
    "reviewed_at",
    "notes",
    "fraud_type",
]

CASE_STATUSES = {"open", "in_review", "closed"}
DETECTION_SOURCES = {"model_flagged", "human_discovered"}
DECISIONS = {
    "true_positive",
    "false_positive",
    "needs_more_review",
    "insufficient_evidence",
}
TRAINABLE_DECISIONS = {"true_positive", "false_positive"}
UNRESOLVED_DECISIONS = {"needs_more_review", "insufficient_evidence"}
FRAUD_TYPES = {"multi_accounting", "payment_fraud", "betting_anomaly", "other"}


class FeedbackStore(Protocol):
    """Storage contract for cases and reviews.

    Loop code depends on these methods, not on SQLite. The later Mongo backend
    should satisfy this same protocol and leave the surrounding logic unchanged.
    """

    def write_cases(self, cases: pd.DataFrame) -> None:
        """Persist newly opened cases."""

    def read_cases(self) -> pd.DataFrame:
        """Return all stored cases."""

    def write_review(self, review: dict[str, Any]) -> None:
        """Append one human review event."""

    def update_case_status(self, case_id: str, status: str) -> None:
        """Update the queue status on the parent case."""

    def read_reviews(self) -> pd.DataFrame:
        """Return all stored review events."""


@dataclass(frozen=True)
class FeedbackRunResult:
    """Outputs from one simulated feedback-loop run."""

    cases: pd.DataFrame
    reviews: pd.DataFrame
    labels: pd.DataFrame
    all_reviewed_labels: pd.DataFrame
    run_manifest: dict[str, Any]
    output_paths: dict[str, Path]


class LocalFeedbackStore:
    """SQLite-backed local store for Phase 5 cases and reviews."""

    def __init__(self, sqlite_path: Path | str):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def write_cases(self, cases: pd.DataFrame) -> None:
        """Insert cases without overwriting an existing case event."""
        if cases.empty:
            return
        prepared = _prepare_case_frame(cases)
        placeholders = ", ".join(["?"] * len(CASE_COLUMNS))
        columns = ", ".join(CASE_COLUMNS)
        with self._connect() as conn:
            conn.executemany(
                f"INSERT OR IGNORE INTO fraud_cases ({columns}) VALUES ({placeholders})",
                prepared[CASE_COLUMNS].itertuples(index=False, name=None),
            )

    def read_cases(self) -> pd.DataFrame:
        """Read all cases from SQLite."""
        with self._connect() as conn:
            frame = pd.read_sql_query("SELECT * FROM fraud_cases", conn)
        if frame.empty:
            return pd.DataFrame(columns=CASE_COLUMNS)
        return _restore_bool_columns(frame, ["model_output_available", "model_routed"])

    def write_review(self, review: dict[str, Any]) -> None:
        """Append one review; existing reviews are never modified."""
        prepared = _prepare_review(review)
        placeholders = ", ".join(["?"] * len(REVIEW_COLUMNS))
        columns = ", ".join(REVIEW_COLUMNS)
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO fraud_reviews ({columns}) VALUES ({placeholders})",
                tuple(prepared[column] for column in REVIEW_COLUMNS),
            )

    def update_case_status(self, case_id: str, status: str) -> None:
        """Move a case through the local review queue."""
        if status not in CASE_STATUSES:
            raise ValueError(f"Unsupported case status: {status}")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE fraud_cases SET status = ? WHERE case_id = ?",
                (status, case_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown case_id: {case_id}")

    def read_reviews(self) -> pd.DataFrame:
        """Read all append-only reviews from SQLite."""
        with self._connect() as conn:
            frame = pd.read_sql_query("SELECT * FROM fraud_reviews", conn)
        if frame.empty:
            return pd.DataFrame(columns=REVIEW_COLUMNS)
        return frame

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fraud_cases (
                    case_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    player_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    overall_risk REAL,
                    band TEXT,
                    multi_accounting_score REAL,
                    payment_fraud_score REAL,
                    betting_anomaly_score REAL,
                    fired_rules TEXT NOT NULL,
                    unmeasured_features TEXT NOT NULL,
                    feature_snapshot TEXT NOT NULL,
                    status TEXT NOT NULL,
                    ruleset_version TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    detection_source TEXT NOT NULL,
                    model_output_available INTEGER NOT NULL,
                    model_routed INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fraud_reviews (
                    review_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    notes TEXT,
                    fraud_type TEXT,
                    FOREIGN KEY(case_id) REFERENCES fraud_cases(case_id)
                )
                """
            )


def run_simulated_feedback_loop(
    *,
    flags_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    sqlite_path: Path | str | None = None,
    store: FeedbackStore | None = None,
    cfg: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> FeedbackRunResult:
    """Run the local Phase 5 mechanism with simulated reviews.

    This is a mechanism test, not a training-data generator. The simulated
    verdicts only prove that cases, reviews, labels, and lineage wire together.
    """
    cfg = cfg or config.load_config()
    feedback_cfg = cfg.get("feedback", {})
    flags_file = Path(flags_path) if flags_path else config.DATA_DIR / "flags.parquet"
    flags = pd.read_parquet(flags_file)
    ruleset_version = get_ruleset_version(cfg)
    snapshot_id = get_snapshot_id(cfg)
    routing_bands = get_routing_bands(cfg)
    created_at = _now_iso()
    run_id = run_id or make_run_id(ruleset_version, snapshot_id, created_at)

    root = Path(output_dir or feedback_cfg.get("output_dir", config.DATA_DIR / "feedback"))
    sqlite_file = Path(sqlite_path or feedback_cfg.get("sqlite_path", root / "feedback.sqlite"))
    local_store = store or LocalFeedbackStore(sqlite_file)

    model_cases = build_model_flagged_cases(
        flags,
        run_id=run_id,
        ruleset_version=ruleset_version,
        snapshot_id=snapshot_id,
        routing_bands=routing_bands,
        created_at=created_at,
    )
    local_store.write_cases(model_cases)

    human_case = build_human_discovered_case(
        _pick_human_discovered_player(flags, routing_bands),
        flags,
        run_id=run_id,
        ruleset_version=ruleset_version,
        snapshot_id=snapshot_id,
        routing_bands=routing_bands,
        created_at=created_at,
    )
    local_store.write_cases(pd.DataFrame([human_case]))

    reviews_written = _write_simulated_reviews(local_store, model_cases, human_case)
    all_reviewed_all, trainable_labels_all = build_labels(local_store)
    cases_all = local_store.read_cases()
    reviews_all = local_store.read_reviews()
    cases = _filter_run(cases_all, run_id)
    run_case_ids = set(cases["case_id"]) if not cases.empty else set()
    reviews = reviews_all[reviews_all["case_id"].isin(run_case_ids)].reset_index(drop=True)
    all_reviewed = _filter_run(all_reviewed_all, run_id)
    trainable_labels = _filter_run(trainable_labels_all, run_id)

    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    labels_path = run_dir / "labels.parquet"
    manifest_path = run_dir / "run_manifest.json"
    trainable_labels.to_parquet(labels_path, index=False)

    manifest = build_run_manifest(
        run_id=run_id,
        created_at=created_at,
        ruleset_version=ruleset_version,
        snapshot_id=snapshot_id,
        flags_path=flags_file,
        flags=flags,
        cases=cases,
        reviews=reviews,
        all_reviewed_labels=all_reviewed,
        trainable_labels=trainable_labels,
        routing_bands=routing_bands,
        reviews_recorded=reviews_written,
        labels_path=labels_path,
        manifest_path=manifest_path,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return FeedbackRunResult(
        cases=cases,
        reviews=reviews,
        labels=trainable_labels,
        all_reviewed_labels=all_reviewed,
        run_manifest=manifest,
        output_paths={
            "sqlite": sqlite_file,
            "labels": labels_path,
            "run_manifest": manifest_path,
        },
    )


def build_model_flagged_cases(
    flags: pd.DataFrame,
    *,
    run_id: str,
    ruleset_version: str,
    snapshot_id: str,
    routing_bands: set[str],
    created_at: str | None = None,
) -> pd.DataFrame:
    """Create cases only for rows that cross the configured routing gate."""
    created_at = created_at or _now_iso()
    routed = flags[flags["band"].isin(routing_bands)].copy()
    cases = [
        _case_from_flag_row(
            row,
            run_id=run_id,
            ruleset_version=ruleset_version,
            snapshot_id=snapshot_id,
            detection_source="model_flagged",
            model_output_available=True,
            model_routed=True,
            created_at=created_at,
        )
        for row in routed.sort_values("player_key").to_dict(orient="records")
    ]
    return pd.DataFrame(cases, columns=CASE_COLUMNS)


def build_human_discovered_case(
    player_key: str,
    flags: pd.DataFrame,
    *,
    run_id: str,
    ruleset_version: str,
    snapshot_id: str,
    routing_bands: set[str],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create an independently discovered case with contemporaneous model state."""
    created_at = created_at or _now_iso()
    matches = flags[flags["player_key"].astype(str) == str(player_key)]
    if matches.empty:
        return _empty_human_case(
            player_key=str(player_key),
            run_id=run_id,
            ruleset_version=ruleset_version,
            snapshot_id=snapshot_id,
            created_at=created_at,
        )
    row = matches.iloc[0].to_dict()
    model_routed = str(row.get("band")) in routing_bands
    return _case_from_flag_row(
        row,
        run_id=run_id,
        ruleset_version=ruleset_version,
        snapshot_id=snapshot_id,
        detection_source="human_discovered",
        model_output_available=True,
        model_routed=model_routed,
        created_at=created_at,
    )


def record_review(
    store: FeedbackStore,
    *,
    case_id: str,
    decision: str,
    reviewed_at: str | None = None,
    notes: str = "",
    fraud_type: str | None = None,
) -> dict[str, Any]:
    """Append a review and move the parent case status to match the decision."""
    reviewed_at = reviewed_at or _now_iso()
    review = _prepare_review(
        {
            "review_id": review_id_for(case_id, reviewed_at, decision, notes, fraud_type),
            "case_id": case_id,
            "decision": decision,
            "reviewed_at": reviewed_at,
            "notes": notes,
            "fraud_type": fraud_type,
        }
    )
    store.write_review(review)
    next_status = "in_review" if decision in UNRESOLVED_DECISIONS else "closed"
    store.update_case_status(case_id, next_status)
    return review


def build_labels(store: FeedbackStore) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join cases to the latest review and return all-reviewed + trainable rows."""
    cases = store.read_cases()
    reviews = store.read_reviews()
    if cases.empty or reviews.empty:
        empty = _empty_labels()
        return empty, empty.copy()

    reviews = reviews.copy()
    reviews["reviewed_at_sort"] = pd.to_datetime(reviews["reviewed_at"], utc=True)
    latest_reviews = (
        reviews.sort_values(["case_id", "reviewed_at_sort", "review_id"])
        .drop_duplicates("case_id", keep="last")
        .drop(columns=["reviewed_at_sort"])
    )
    labels = cases.merge(latest_reviews, on="case_id", how="inner", validate="one_to_one")
    labels["is_trainable_label"] = labels["decision"].isin(TRAINABLE_DECISIONS)
    labels["is_false_negative"] = labels.apply(is_false_negative_label, axis=1)
    labels = labels.sort_values(["run_id", "player_key", "case_id"], kind="mergesort").reset_index(
        drop=True
    )
    trainable = labels[labels["is_trainable_label"]].reset_index(drop=True)
    return labels, trainable


def build_run_manifest(
    *,
    run_id: str,
    created_at: str,
    ruleset_version: str,
    snapshot_id: str,
    flags_path: Path,
    flags: pd.DataFrame,
    cases: pd.DataFrame,
    reviews: pd.DataFrame,
    all_reviewed_labels: pd.DataFrame,
    trainable_labels: pd.DataFrame,
    routing_bands: set[str],
    reviews_recorded: int,
    labels_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Create MLflow-friendly run metadata without inventing model metrics."""
    return {
        "run_id": run_id,
        "created_at": created_at,
        "ruleset_version": ruleset_version,
        "snapshot_id": snapshot_id,
        "flags_input_path": str(flags_path),
        "flags_row_count": int(len(flags)),
        "routing_bands": sorted(routing_bands),
        "routed_case_count": int((flags["band"].isin(routing_bands)).sum()),
        "case_row_count": int(len(cases)),
        "review_row_count": int(len(reviews)),
        "reviews_recorded_this_run": int(reviews_recorded),
        "reviewed_case_count": int(len(all_reviewed_labels)),
        "trainable_label_count": int(len(trainable_labels)),
        "non_trainable_reviewed_count": int(
            (~all_reviewed_labels["is_trainable_label"]).sum()
            if "is_trainable_label" in all_reviewed_labels
            else 0
        ),
        "human_discovered_case_count": int(
            (cases["detection_source"] == "human_discovered").sum()
            if "detection_source" in cases
            else 0
        ),
        "false_negative_label_count": int(
            trainable_labels["is_false_negative"].sum()
            if "is_false_negative" in trainable_labels
            else 0
        ),
        "artifact_paths": {
            "labels": str(labels_path),
            "run_manifest": str(manifest_path),
        },
        "notes": [
            "Phase 5 local-storage-first plumbing run; simulated reviews are not real labels.",
            "Dev routed model cases are dominated by the known R-MA-03 office-device artifact, not detections.",
            "No precision/recall/label-quality metrics are emitted because no real labels exist.",
        ],
    }


def _filter_run(frame: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Limit a case/label frame to one feedback run."""
    if frame.empty or "run_id" not in frame:
        return frame.copy()
    return frame[frame["run_id"] == run_id].reset_index(drop=True)


def get_routing_bands(cfg: dict[str, Any]) -> set[str]:
    """Configured placeholder bands that create model_flagged cases."""
    return {str(band) for band in cfg.get("feedback", {}).get("routing_bands", ["medium", "high"])}


def get_ruleset_version(cfg: dict[str, Any]) -> str:
    """Return the scoring ruleset version stamped on each opened case."""
    version = cfg.get("scoring", {}).get("ruleset_version")
    if not version:
        raise ValueError("config.yaml scoring.ruleset_version is required for feedback lineage.")
    return str(version)


def get_snapshot_id(cfg: dict[str, Any]) -> str:
    """Derive snapshot lineage from the frozen Phase 3 input directory."""
    input_dir = cfg.get("phase3", {}).get("input_dir")
    if not input_dir:
        raise ValueError("config.yaml phase3.input_dir is required for feedback lineage.")
    return Path(str(input_dir)).name


def make_run_id(ruleset_version: str, snapshot_id: str, created_at: str | None = None) -> str:
    """Create a path-safe run ID from timestamp, ruleset, and snapshot."""
    stamp = created_at or _now_iso()
    compact_stamp = re.sub(r"[^0-9TZ]", "", stamp).replace("Z", "Z")
    return _safe_id(f"feedback_{compact_stamp}_{ruleset_version}_{snapshot_id}")


def case_id_for(run_id: str, player_key: str, detection_source: str) -> str:
    """Deterministic case key for one player in one feedback/scoring run."""
    _validate_detection_source(detection_source)
    digest = hashlib.sha256(f"{run_id}|{player_key}|{detection_source}".encode("utf-8")).hexdigest()
    return f"case_{digest[:20]}"


def review_id_for(
    case_id: str,
    reviewed_at: str,
    decision: str,
    notes: str = "",
    fraud_type: str | None = None,
) -> str:
    """Deterministic review key for an append-only decision event."""
    digest = hashlib.sha256(
        f"{case_id}|{reviewed_at}|{decision}|{notes}|{fraud_type or ''}".encode("utf-8")
    ).hexdigest()
    return f"review_{digest[:20]}"


def is_false_negative_label(row: pd.Series | dict[str, Any]) -> bool:
    """A false negative is inferred, never selected as a review decision."""
    return (
        row.get("decision") == "true_positive"
        and row.get("detection_source") == "human_discovered"
        and not bool(row.get("model_routed"))
    )


def _write_simulated_reviews(
    store: FeedbackStore,
    model_cases: pd.DataFrame,
    human_case: dict[str, Any],
) -> int:
    """Write a small deterministic set of fixture reviews for the phase gate."""
    written = 0
    if not model_cases.empty:
        first = model_cases.iloc[0]
        record_review(
            store,
            case_id=str(first["case_id"]),
            decision="false_positive",
            reviewed_at="2026-06-25T00:01:00Z",
            notes="Simulated review: dev route is known device-artifact plumbing.",
            fraud_type=None,
        )
        written += 1
    if len(model_cases) > 1:
        second = model_cases.iloc[1]
        record_review(
            store,
            case_id=str(second["case_id"]),
            decision="needs_more_review",
            reviewed_at="2026-06-25T00:02:00Z",
            notes="Simulated unresolved review to exercise non-trainable queue handling.",
            fraud_type=None,
        )
        written += 1
    if len(model_cases) > 2:
        third = model_cases.iloc[2]
        record_review(
            store,
            case_id=str(third["case_id"]),
            decision="insufficient_evidence",
            reviewed_at="2026-06-25T00:03:00Z",
            notes="Simulated unresolved review to exercise exclusion from trainable labels.",
            fraud_type=None,
        )
        written += 1
    record_review(
        store,
        case_id=str(human_case["case_id"]),
        decision="true_positive",
        reviewed_at="2026-06-25T00:04:00Z",
        notes="Simulated human-discovered miss; proves false negatives are emergent.",
        fraud_type="multi_accounting",
    )
    return written + 1


def _pick_human_discovered_player(flags: pd.DataFrame, routing_bands: set[str]) -> str:
    """Choose a non-routed player so the simulated run exercises Channel 2."""
    non_routed = flags[~flags["band"].isin(routing_bands)].sort_values("player_key")
    if non_routed.empty:
        return "__simulated_unscored_player__"
    return str(non_routed.iloc[0]["player_key"])


def _case_from_flag_row(
    row: dict[str, Any],
    *,
    run_id: str,
    ruleset_version: str,
    snapshot_id: str,
    detection_source: str,
    model_output_available: bool,
    model_routed: bool,
    created_at: str,
) -> dict[str, Any]:
    player_key = str(row["player_key"])
    _validate_detection_source(detection_source)
    return {
        "case_id": case_id_for(run_id, player_key, detection_source),
        "run_id": run_id,
        "player_key": player_key,
        "created_at": created_at,
        "overall_risk": _nullable_float(row.get("overall_risk")),
        "band": None if pd.isna(row.get("band")) else str(row.get("band")),
        "multi_accounting_score": _nullable_float(row.get("multi_accounting_score")),
        "payment_fraud_score": _nullable_float(row.get("payment_fraud_score")),
        "betting_anomaly_score": _nullable_float(row.get("betting_anomaly_score")),
        "fired_rules": _json_text(row.get("fired_rules"), default=[]),
        "unmeasured_features": _json_text(row.get("unmeasured_features"), default=[]),
        "feature_snapshot": _json_text(row.get("feature_snapshot"), default={}),
        "status": "open",
        "ruleset_version": ruleset_version,
        "snapshot_id": snapshot_id,
        "detection_source": detection_source,
        "model_output_available": bool(model_output_available),
        "model_routed": bool(model_routed),
    }


def _empty_human_case(
    *,
    player_key: str,
    run_id: str,
    ruleset_version: str,
    snapshot_id: str,
    created_at: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id_for(run_id, player_key, "human_discovered"),
        "run_id": run_id,
        "player_key": player_key,
        "created_at": created_at,
        "overall_risk": None,
        "band": None,
        "multi_accounting_score": None,
        "payment_fraud_score": None,
        "betting_anomaly_score": None,
        "fired_rules": "[]",
        "unmeasured_features": "[]",
        "feature_snapshot": "{}",
        "status": "open",
        "ruleset_version": ruleset_version,
        "snapshot_id": snapshot_id,
        "detection_source": "human_discovered",
        "model_output_available": False,
        "model_routed": False,
    }


def _prepare_case_frame(cases: pd.DataFrame) -> pd.DataFrame:
    prepared = cases.copy()
    for column in CASE_COLUMNS:
        if column not in prepared:
            prepared[column] = None
    for source in prepared["detection_source"]:
        _validate_detection_source(str(source))
    for status in prepared["status"]:
        if status not in CASE_STATUSES:
            raise ValueError(f"Unsupported case status: {status}")
    for column in ["fired_rules", "unmeasured_features", "feature_snapshot"]:
        prepared[column] = prepared[column].map(_json_text)
    prepared["model_output_available"] = prepared["model_output_available"].map(lambda value: int(bool(value)))
    prepared["model_routed"] = prepared["model_routed"].map(lambda value: int(bool(value)))
    return prepared[CASE_COLUMNS]


def _prepare_review(review: dict[str, Any]) -> dict[str, Any]:
    decision = str(review["decision"])
    if decision not in DECISIONS:
        raise ValueError(f"Unsupported review decision: {decision}")
    fraud_type = review.get("fraud_type")
    if fraud_type is not None and fraud_type not in FRAUD_TYPES:
        raise ValueError(f"Unsupported fraud_type: {fraud_type}")
    return {
        "review_id": str(review["review_id"]),
        "case_id": str(review["case_id"]),
        "decision": decision,
        "reviewed_at": str(review["reviewed_at"]),
        "notes": "" if review.get("notes") is None else str(review.get("notes")),
        "fraud_type": fraud_type,
    }


def _restore_bool_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    restored = frame.copy()
    for column in columns:
        restored[column] = restored[column].astype(bool)
    return restored


def _empty_labels() -> pd.DataFrame:
    label_columns = CASE_COLUMNS + REVIEW_COLUMNS + ["is_trainable_label", "is_false_negative"]
    return pd.DataFrame(columns=label_columns)


def _validate_detection_source(detection_source: str) -> None:
    if detection_source not in DETECTION_SOURCES:
        raise ValueError(f"Unsupported detection_source: {detection_source}")


def _json_text(value: Any, default: Any | None = None) -> str:
    if value is None or (not isinstance(value, (list, dict, str)) and pd.isna(value)):
        value = default if default is not None else []
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    result = run_simulated_feedback_loop()
    print(json.dumps(result.run_manifest, indent=2, sort_keys=True))
