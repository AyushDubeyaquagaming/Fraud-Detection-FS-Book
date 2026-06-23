"""Typed feature result contract shared by all feature groups.

Every feature returns the same five pieces: the scalar value, reviewer evidence,
why it is null if it cannot be measured, its strength, and how scoring should
use it. Keeping this small contract strict prevents silent null/zero confusion
later in Phase 4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


FeatureStrength = Literal["strong", "moderate", "weak", "context_only"]
FeatureScoringRole = Literal["scoring", "supporting", "context_only"]
FeatureValue = int | float | bool | None
FeatureEvidence = list[dict[str, Any]]


@dataclass(frozen=True)
class FeatureResult:
    """One player's scalar, evidence, null state, and scoring metadata.

    `None` means "we cannot measure this feature for this player" and must
    carry a reason. A real zero means "we measured it and found none." Non-zero
    values must include evidence so a reviewer can inspect why the feature fired.
    """

    feature_value: FeatureValue
    feature_evidence: FeatureEvidence
    feature_null_reason: str | None
    feature_strength: FeatureStrength
    feature_scoring_role: FeatureScoringRole

    def __post_init__(self) -> None:
        """Enforce the null contract at the point each feature is created."""
        if self.feature_value is None and not self.feature_null_reason:
            raise ValueError("Null feature values require feature_null_reason.")
        if self.feature_value is not None and self.feature_null_reason is not None:
            raise ValueError("Measured feature values cannot have feature_null_reason.")
        if self.feature_value is not None and bool(self.feature_value) and not self.feature_evidence:
            raise ValueError("Non-zero feature values require reviewer evidence.")
