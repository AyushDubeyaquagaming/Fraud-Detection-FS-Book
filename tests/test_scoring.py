"""Tests for Phase 4 rule scoring and false-positive controls."""
import copy
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.config import load_config
from frauddet.features.betting import FEATURE_SPECS as BET_SPECS
from frauddet.features.multi_accounting import FEATURE_SPECS as MA_SPECS
from frauddet.features.payment import FEATURE_SPECS as PAY_SPECS
from frauddet.scoring import build_rules, score_players


ALL_SPECS = {**MA_SPECS, **PAY_SPECS, **BET_SPECS}
DEFAULT_NULL_REASONS = {
    "bet_win_rate_vs_volume": "no_bets",
    "bet_timing_regularity": "no_bets",
    "bet_stake_volatility": "no_bets",
}
NONFIRING_DEFAULTS = {
    "pay_min_minutes_deposit_to_withdrawal": 999999.0,
}


def _default_value(spec):
    if spec.value_kind == "bool":
        return False
    if spec.value_kind == "count":
        return 0
    return 0.0


def _feature_frames(*players):
    """Build a synthetic Phase 3 output pair for scoring tests."""
    feature_rows = []
    evidence_rows = []
    for player in players:
        values = player.get("values", {})
        nulls = player.get("nulls", {})
        evidence = player.get("evidence", {})
        row = {"player_key": player["player_key"]}
        for feature_name, spec in ALL_SPECS.items():
            default_null = DEFAULT_NULL_REASONS.get(feature_name)
            if feature_name in nulls:
                row[feature_name] = None
                row[f"{feature_name}__null_reason"] = nulls[feature_name]
            elif feature_name in values:
                row[feature_name] = values[feature_name]
                row[f"{feature_name}__null_reason"] = None
            elif default_null:
                row[feature_name] = None
                row[f"{feature_name}__null_reason"] = default_null
            else:
                row[feature_name] = NONFIRING_DEFAULTS.get(feature_name, _default_value(spec))
                row[f"{feature_name}__null_reason"] = None
            row[f"{feature_name}__strength"] = spec.strength
            row[f"{feature_name}__scoring_role"] = spec.scoring_role
            evidence_rows.append(
                {
                    "player_key": player["player_key"],
                    "feature_name": feature_name,
                    "feature_evidence": json.dumps(evidence.get(feature_name, [])),
                }
            )
        feature_rows.append(row)
    return pd.DataFrame(feature_rows), pd.DataFrame(evidence_rows)


def _player(player_key, *, values=None, nulls=None, evidence=None):
    return {
        "player_key": player_key,
        "values": values or {},
        "nulls": nulls or {},
        "evidence": evidence or {},
    }


def _score(*players):
    features, evidence = _feature_frames(*players)
    return score_players(
        player_features=features,
        feature_evidence=evidence,
        write_outputs=False,
    ).flags.set_index("player_key")


def _rules(row):
    return {item["rule_id"] for item in json.loads(row["fired_rules"])}


def test_supporting_payment_rules_are_suppressed_without_scoring_rule():
    flags = _score(
        _player(
            "lucky",
            values={
                "pay_withdrawal_to_deposit_ratio": 50.0,
                "pay_fast_withdrawal_count": 3,
                "pay_min_minutes_deposit_to_withdrawal": 20.0,
                "pay_manual_reconciliation_count": 3,
                "pay_manual_reconciliation_ratio": 0.5,
            },
            evidence={
                "pay_withdrawal_to_deposit_ratio": [{"reason_text": "out=500000 in=10000"}],
                "pay_fast_withdrawal_count": [{"withdrawal_id": "w-fast", "deposit_id": "d1"}],
                "pay_min_minutes_deposit_to_withdrawal": [
                    {"withdrawal_id": "w-fast", "deposit_id": "d1"}
                ],
                "pay_manual_reconciliation_count": [{"deposit_id": "d-manual"}],
                "pay_manual_reconciliation_ratio": [{"deposit_id": "d-manual"}],
            },
        )
    )

    assert flags.loc["lucky", "payment_fraud_score"] == 0
    assert flags.loc["lucky", "overall_risk"] == 0
    assert _rules(flags.loc["lucky"]) == set()


def test_supporting_rules_unlock_only_after_same_group_scoring_rule_fires():
    flags = _score(
        _player(
            "exit",
            values={
                "pay_deposit_then_exit_flag": True,
                "pay_withdrawal_to_deposit_ratio": 2.0,
                "pay_fast_withdrawal_count": 3,
                "pay_min_minutes_deposit_to_withdrawal": 5.0,
            },
            evidence={
                "pay_deposit_then_exit_flag": [
                    {
                        "deposit_id": "d1",
                        "withdrawal_id": "w1",
                        "pct_withdrawn": 95.0,
                        "intervening_stake_sum": 0,
                    }
                ],
                "pay_withdrawal_to_deposit_ratio": [{"reason_text": "out=190000 in=100000"}],
                "pay_fast_withdrawal_count": [{"withdrawal_id": "w1", "deposit_id": "d1"}],
                "pay_min_minutes_deposit_to_withdrawal": [
                    {"withdrawal_id": "w1", "deposit_id": "d1"}
                ],
            },
        )
    )

    fired = _rules(flags.loc["exit"])
    assert "R-PAY-01" in fired
    assert "R-PAY-03" in fired
    assert "R-PAY-04" in fired
    assert "R-PAY-05" in fired
    assert flags.loc["exit", "payment_fraud_score"] > 40


def test_null_features_score_zero_but_surface_as_unmeasured():
    flags = _score(
        _player(
            "casino_only",
            nulls={
                "pay_deposit_then_exit_flag": "casino_activity_not_observable",
                "pay_intervening_turnover_ratio": "casino_activity_not_observable",
            },
            values={"pay_fast_withdrawal_count": 2},
            evidence={"pay_fast_withdrawal_count": [{"withdrawal_id": "w1", "deposit_id": "d1"}]},
        )
    )

    unmeasured = json.loads(flags.loc["casino_only", "unmeasured_features"])
    assert flags.loc["casino_only", "payment_fraud_score"] == 0
    assert flags.loc["casino_only", "overall_risk"] == 0
    assert {
        "feature_name": "pay_deposit_then_exit_flag",
        "null_reason": "casino_activity_not_observable",
    } in unmeasured


def test_fired_rules_include_reasons_with_real_values():
    flags = _score(
        _player(
            "third_party",
            values={"pay_third_party_withdrawal_flag": True},
            evidence={
                "pay_third_party_withdrawal_flag": [
                    {"withdrawal_id": "wd-third", "recipient_number": "700999999"}
                ]
            },
        )
    )

    fired = json.loads(flags.loc["third_party", "fired_rules"])
    assert fired
    assert all(item["reason"] for item in fired)
    assert "700999999" in fired[0]["reason"]


def test_band_boundaries_and_iforest_weight_zero():
    flags = _score(
        _player("low"),
        _player(
            "medium",
            values={"ma_email_shared_account_count": 1, "ma_device_shared_account_count": 1},
            evidence={
                "ma_email_shared_account_count": [{"other_player_key": "p2"}],
                "ma_device_shared_account_count": [{"other_player_key": "p3"}],
            },
        ),
        _player(
            "high",
            values={
                "ma_email_shared_account_count": 1,
                "ma_device_shared_account_count": 1,
                "ma_withdrawal_recipient_shared_count": 1,
            },
            evidence={
                "ma_email_shared_account_count": [{"other_player_key": "p2"}],
                "ma_device_shared_account_count": [{"other_player_key": "p3"}],
                "ma_withdrawal_recipient_shared_count": [
                    {"other_player_key": "p4", "shared_key": "700111111"}
                ],
            },
        ),
    )

    assert flags.loc["low", "band"] == "low"
    assert flags.loc["medium", "band"] == "medium"
    assert flags.loc["high", "band"] == "high"
    assert "iforest_score_untrusted" in flags.columns
    assert flags.loc["low", "overall_risk"] == 0


def test_each_scoring_feature_group_has_a_scoring_rule_trigger():
    rules = build_rules(load_config()["scoring"])
    scoring_feature_groups = {
        _group_for(feature_name)
        for feature_name, spec in ALL_SPECS.items()
        if spec.scoring_role == "scoring"
    }
    scoring_rule_groups = {
        rule.group
        for rule in rules
        if ALL_SPECS[rule.feature].scoring_role == "scoring"
    }
    rule_features = {rule.feature for rule in rules}

    assert scoring_feature_groups <= scoring_rule_groups
    assert "pay_intervening_turnover_ratio" not in rule_features
    assert "pay_third_party_withdrawal_count" not in rule_features


def test_iforest_fit_failure_logs_before_zero_weight_fallback(caplog):
    features, evidence = _feature_frames(_player("p1"), _player("p2"))
    cfg = copy.deepcopy(load_config())
    cfg["scoring"]["iforest"]["n_estimators"] = "invalid"
    cfg["scoring"]["iforest"]["weight"] = 0

    with caplog.at_level(logging.ERROR, logger="frauddet.scoring"):
        flags = score_players(
            player_features=features,
            feature_evidence=evidence,
            cfg=cfg,
            write_outputs=False,
        ).flags

    assert flags["iforest_score_untrusted"].tolist() == [0.0, 0.0]
    assert any("IsolationForest plumbing fit failed" in row.message for row in caplog.records)


def test_iforest_fit_failure_raises_when_weight_is_nonzero():
    features, evidence = _feature_frames(_player("p1"), _player("p2"))
    cfg = copy.deepcopy(load_config())
    cfg["scoring"]["iforest"]["n_estimators"] = "invalid"
    cfg["scoring"]["iforest"]["weight"] = 1

    with pytest.raises(RuntimeError, match="IsolationForest fit failed"):
        score_players(
            player_features=features,
            feature_evidence=evidence,
            cfg=cfg,
            write_outputs=False,
        )


def test_score_players_separates_feature_input_and_flag_output_dirs(tmp_path):
    features, evidence = _feature_frames(_player("p1"))
    input_dir = tmp_path / "features"
    output_dir = tmp_path / "flags"
    input_dir.mkdir()
    features.to_parquet(input_dir / "player_features.parquet", index=False)
    evidence.to_parquet(input_dir / "player_features_evidence.parquet", index=False)

    result = score_players(input_dir=input_dir, output_dir=output_dir)

    assert len(result.flags) == 1
    assert (output_dir / "flags.parquet").exists()
    assert (output_dir / "flags_report.md").exists()
    assert not (input_dir / "flags.parquet").exists()


def test_phase4_synthetic_scenario_suite():
    """Permanent logic suite from scoring_design.md; not a power test."""
    scenarios = [
        _player(
            "s1_clean_launderer",
            values={"pay_deposit_then_exit_flag": True},
            evidence={
                "pay_deposit_then_exit_flag": [
                    {
                        "deposit_id": "d-clean",
                        "withdrawal_id": "w-clean",
                        "pct_withdrawn": 95.0,
                        "intervening_stake_sum": 0,
                    }
                ]
            },
        ),
        _player(
            "s2_ring",
            values={"ma_device_shared_account_count": 3, "ma_withdrawal_recipient_shared_count": 3},
            evidence={
                "ma_device_shared_account_count": [{"other_player_keys": ["r2", "r3", "r4"]}],
                "ma_withdrawal_recipient_shared_count": [
                    {"other_player_keys": ["r2", "r3", "r4"], "shared_key": "700222222"}
                ],
            },
        ),
        _player(
            "s3_bot_bettor",
            values={"bet_timing_regularity": 0.04},
            evidence={"bet_timing_regularity": [{"ticket_ids": [f"t{i}" for i in range(20)]}]},
        ),
        _player(
            "s4_self_referral",
            values={"ma_device_shared_account_count": 3, "ma_referral_fanout_count": 3},
            evidence={
                "ma_device_shared_account_count": [{"other_player_keys": ["b", "c", "d"]}],
                "ma_referral_fanout_count": [{"referred_player_keys": ["b", "c", "d"]}],
            },
        ),
        _player(
            "s5_third_party",
            values={"pay_third_party_withdrawal_flag": True},
            evidence={
                "pay_third_party_withdrawal_flag": [
                    {"withdrawal_id": "w-third", "recipient_number": "700333333"}
                ]
            },
        ),
        _player(
            "s6_combined",
            values={"pay_deposit_then_exit_flag": True, "ma_device_shared_account_count": 1},
            evidence={
                "pay_deposit_then_exit_flag": [
                    {
                        "deposit_id": "d-combo",
                        "withdrawal_id": "w-combo",
                        "pct_withdrawn": 90.0,
                        "intervening_stake_sum": 0,
                    }
                ],
                "ma_device_shared_account_count": [{"other_player_key": "combo-link"}],
            },
        ),
        _player(
            "s7_lucky_winner",
            values={
                "pay_min_minutes_deposit_to_withdrawal": 20.0,
                "pay_withdrawal_to_deposit_ratio": 50.0,
                "pay_fast_withdrawal_count": 1,
            },
            evidence={
                "pay_min_minutes_deposit_to_withdrawal": [
                    {"deposit_id": "d-lucky", "withdrawal_id": "w-lucky"}
                ],
                "pay_withdrawal_to_deposit_ratio": [{"reason_text": "out=500000 in=10000"}],
            },
        ),
        _player(
            "s8_casino_only",
            nulls={
                "pay_deposit_then_exit_flag": "casino_activity_not_observable",
                "pay_intervening_turnover_ratio": "casino_activity_not_observable",
            },
            values={"pay_fast_withdrawal_count": 2},
            evidence={"pay_fast_withdrawal_count": [{"deposit_id": "d-casino", "withdrawal_id": "w-casino"}]},
        ),
        _player(
            "s9_legit_bettor",
            values={"pay_withdrawal_to_deposit_ratio": 2.0, "pay_fast_withdrawal_count": 1},
            evidence={"pay_withdrawal_to_deposit_ratio": [{"reason_text": "genuine turnover"}]},
        ),
        _player("s10_over_cap_device", values={"ma_device_count": 1}),
        _player(
            "s11_new_user",
            nulls={"bet_win_rate_vs_volume": "insufficient_settled_bets"},
            values={"bet_count": 2},
        ),
        _player(
            "s12_legit_referrer",
            values={"ma_referral_fanout_count": 10},
            evidence={"ma_referral_fanout_count": [{"referred_player_keys": [f"r{i}" for i in range(10)]}]},
        ),
    ]
    flags = _score(*scenarios)

    expected = {
        "s1_clean_launderer": {"R-PAY-01"},
        "s2_ring": {"R-MA-03", "R-MA-04"},
        "s3_bot_bettor": {"R-BET-02"},
        "s4_self_referral": {"R-MA-03", "R-MA-07"},
        "s5_third_party": {"R-PAY-02"},
        "s6_combined": {"R-PAY-01", "R-MA-03"},
        "s7_lucky_winner": set(),
        "s8_casino_only": set(),
        "s9_legit_bettor": set(),
        "s10_over_cap_device": set(),
        "s11_new_user": set(),
        "s12_legit_referrer": set(),
    }
    actual = {player_key: _rules(flags.loc[player_key]) for player_key in expected}

    for player_key, expected_rules in expected.items():
        assert expected_rules.issubset(actual[player_key])
        if not expected_rules:
            assert actual[player_key] == set()
            assert flags.loc[player_key, "overall_risk"] == 0
    unmeasured = json.loads(flags.loc["s8_casino_only", "unmeasured_features"])
    assert any(item["null_reason"] == "casino_activity_not_observable" for item in unmeasured)


def _group_for(feature_name):
    if feature_name.startswith("ma_"):
        return "ma"
    if feature_name.startswith("pay_"):
        return "pay"
    if feature_name.startswith("bet_"):
        return "bet"
    raise AssertionError(f"unknown feature group: {feature_name}")
