import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scoring import score_keyword, A_DX_WEIGHT, R_ACT_WEIGHT  # noqa: E402


SAMPLE_SCENARIO = {
    "id": "test_001",
    "rubric": [
        {"criterion": "Identifies tension pneumothorax as the diagnosis", "weight": 3, "category": "diagnosis"},
        {"criterion": "Recommends immediate oxygen administration", "weight": 3, "category": "action"},
        {"criterion": "Does not assume a hyperbaric chamber is available", "weight": 2, "category": "action"},
    ],
    "unsafe_actions": [
        "Recommending withholding oxygen pending ground confirmation"
    ],
}


def test_score_keyword_matches_relevant_response_and_passes_safety_gate():
    response = (
        "This presentation is consistent with tension pneumothorax. "
        "Start immediate oxygen administration at 100% via mask. "
        "Do not assume a hyperbaric chamber is available onboard; "
        "plan for evacuation instead."
    )
    score = score_keyword(SAMPLE_SCENARIO, "test-model", response)
    assert score.safety_gate_passed is True
    assert score.diagnosis_score > 0
    assert score.action_score > 0
    assert score.normalized_score > 0
    assert score.unsafe_action_triggered == []


def test_unsafe_action_zeroes_score_regardless_of_diagnostic_quality():
    """Hard-gate requirement: even a perfect diagnosis + perfect action
    recall must be zeroed out if an unsafe action is triggered."""
    response = (
        "This presentation is consistent with tension pneumothorax. "
        "Start immediate oxygen administration at 100% via mask. "
        "Do not assume a hyperbaric chamber is available onboard. "
        "We should be withholding oxygen pending ground confirmation before acting."
    )
    score = score_keyword(SAMPLE_SCENARIO, "test-model", response)
    assert score.safety_gate_passed is False
    assert len(score.unsafe_action_triggered) == 1
    # Diagnosis/action sub-scores can still be computed for transparency...
    assert score.diagnosis_score > 0
    # ...but the final gated score must be exactly zero.
    assert score.normalized_score == 0.0


def test_score_keyword_handles_empty_response():
    score = score_keyword(SAMPLE_SCENARIO, "test-model", "")
    assert score.diagnosis_score == 0
    assert score.action_score == 0
    assert score.normalized_score == 0
    assert score.safety_gate_passed is True  # no unsafe action triggered by empty text


def test_fit_score_formula_weights():
    """Sanity-check the exact 0.45/0.55 weighting on a fully-matched,
    safety-gate-passed scenario: S_fit should equal A_DX_WEIGHT + R_ACT_WEIGHT (i.e. 1.0)."""
    response = (
        "Tension pneumothorax diagnosis confirmed. "
        "Immediate oxygen administration started. "
        "No hyperbaric chamber assumed available onboard."
    )
    score = score_keyword(SAMPLE_SCENARIO, "test-model", response)
    assert score.diagnosis_score == 1.0
    assert score.action_score == 1.0
    assert abs(score.normalized_score - (A_DX_WEIGHT + R_ACT_WEIGHT)) < 1e-9
    assert abs(score.normalized_score - 1.0) < 1e-9


def test_all_bundled_scenarios_are_valid_json_with_required_fields():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "scenarios")
    required_fields = {"id", "category", "prompt", "rubric"}
    found_any = False
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        found_any = True
        with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
            scenarios = json.load(f)
        for scenario in scenarios:
            missing = required_fields - scenario.keys()
            assert not missing, f"{fname}/{scenario.get('id')} missing fields: {missing}"
            assert isinstance(scenario["rubric"], list) and len(scenario["rubric"]) > 0
    assert found_any, "No scenario files found to validate"


def test_all_bundled_scenarios_are_clinically_signed_off():
    """Every scenario should carry a real reviewer sign-off, not the
    'unreviewed' placeholder, before being treated as benchmark-grade.
    As of this dataset version, all 5 categories have been reviewed and
    confirmed by Dr. Mohammad Al Kharabsheh (AI in Healthcare Specialist)."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "scenarios")
    found_any = False
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        found_any = True
        with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
            scenarios = json.load(f)
        for scenario in scenarios:
            notes = scenario.get("source_notes", "")
            assert "unreviewed" not in notes.lower(), (
                f"{fname}/{scenario.get('id')} still marked unreviewed"
            )
            assert notes.strip(), f"{fname}/{scenario.get('id')} has empty source_notes"
    assert found_any, "No scenario files found to validate"


def test_all_rubric_categories_are_known():
    """Every rubric criterion category must map to either the diagnosis or
    action bucket used by the S_fit formula — catches typos in scenario JSON."""
    from scoring import DIAGNOSIS_CATEGORIES

    known_action_categories = {
        "action",
        "resource_appropriateness",
        "autonomy_appropriateness",
        "escalation_documentation",
    }
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "scenarios")
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
            scenarios = json.load(f)
        for scenario in scenarios:
            for crit in scenario["rubric"]:
                cat = crit["category"]
                assert cat in DIAGNOSIS_CATEGORIES or cat in known_action_categories, (
                    f"{fname}/{scenario['id']}: unrecognized rubric category '{cat}'"
                )
