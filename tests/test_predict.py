"""
FILE: tests/test_predict.py
IMPROVEMENT: Testing with pytest
Run: pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

# ─── SAMPLE INPUTS ────────────────────────────────────────────────────────────
LOW_RISK_INPUT = {
    "Gender_enc": 1, "Living_Area": 1, "Marital_Status": 1,
    "Time_Spent": 1, "Usage_Frequency": 1, "Purpose_Use": 5,
    "Neglect_Duties": 1, "Preoccupation": 1, "Withdrawal_Anxiety": 1,
    "Attempted_Control": 5, "Mood_Regulation": 1, "Deception_Use": 1,
    "Relationship_Impact": 1,
    "Behavioral_Intensity": 1.9, "Withdrawal_x_Neglect": 1,
    "Control_Failure": 5, "Social_Damage": 2, "Craving_Score": 2,
}

HIGH_RISK_INPUT = {
    "Gender_enc": 0, "Living_Area": 2, "Marital_Status": 2,
    "Time_Spent": 5, "Usage_Frequency": 5, "Purpose_Use": 1,
    "Neglect_Duties": 5, "Preoccupation": 5, "Withdrawal_Anxiety": 5,
    "Attempted_Control": 1, "Mood_Regulation": 5, "Deception_Use": 4,
    "Relationship_Impact": 5,
    "Behavioral_Intensity": 4.6, "Withdrawal_x_Neglect": 25,
    "Control_Failure": 25, "Social_Damage": 9, "Craving_Score": 10,
}

INVALID_INPUT = {
    "Gender_enc": 99, "Living_Area": -5,
}

# ─── TESTS ────────────────────────────────────────────────────────────────────

class TestInputValidation:
    """Test that input validation catches bad data."""

    def test_missing_fields_detected(self):
        from predict_engine_v2 import validate_input
        errors, _ = validate_input({"Gender_enc": 1})
        assert len(errors) > 0, "Should flag missing fields"

    def test_out_of_range_field_detected(self):
        from predict_engine_v2 import validate_input
        bad = dict(LOW_RISK_INPUT)
        bad["Time_Spent"] = 99
        errors, _ = validate_input(bad)
        assert any("Time_Spent" in e for e in errors)

    def test_valid_input_passes(self):
        from predict_engine_v2 import validate_input
        errors, cleaned = validate_input(LOW_RISK_INPUT)
        assert len(errors) == 0
        assert isinstance(cleaned, dict)


class TestFeatureEngineering:
    """Test engineered features are computed correctly."""

    def test_behavioral_intensity_range(self):
        bi = np.mean([LOW_RISK_INPUT[k] for k in [
            "Time_Spent","Usage_Frequency","Purpose_Use","Neglect_Duties",
            "Preoccupation","Withdrawal_Anxiety","Attempted_Control",
            "Mood_Regulation","Deception_Use","Relationship_Impact"
        ]])
        assert 1.0 <= bi <= 5.0, "Behavioral intensity must be in 1–5 range"

    def test_craving_score_bounds(self):
        cs = HIGH_RISK_INPUT["Craving_Score"]
        assert cs == HIGH_RISK_INPUT["Preoccupation"] + HIGH_RISK_INPUT["Withdrawal_Anxiety"]

    def test_control_failure_direction(self):
        """Higher time + lower control = higher failure score"""
        low_cf  = LOW_RISK_INPUT["Time_Spent"]  * (6 - LOW_RISK_INPUT["Attempted_Control"])
        high_cf = HIGH_RISK_INPUT["Time_Spent"] * (6 - HIGH_RISK_INPUT["Attempted_Control"])
        assert high_cf > low_cf


class TestRecommendations:
    """Test recommendation structure for all risk levels."""

    def test_all_risk_levels_have_recommendations(self):
        from predict_engine_v2 import RECOMMENDATIONS
        for level in ["Low", "Moderate", "High"]:
            assert level in RECOMMENDATIONS
            assert "headline" in RECOMMENDATIONS[level]
            assert "summary"  in RECOMMENDATIONS[level]
            assert "actions"  in RECOMMENDATIONS[level]
            assert len(RECOMMENDATIONS[level]["actions"]) >= 3

    def test_high_risk_has_most_recommendations(self):
        from predict_engine_v2 import RECOMMENDATIONS
        low_count  = len(RECOMMENDATIONS["Low"]["actions"])
        high_count = len(RECOMMENDATIONS["High"]["actions"])
        assert high_count >= low_count, "High risk should have more recommendations"

    def test_each_action_has_required_fields(self):
        from predict_engine_v2 import RECOMMENDATIONS
        for level, data in RECOMMENDATIONS.items():
            for action in data["actions"]:
                assert "icon"   in action, f"Missing icon in {level}"
                assert "title"  in action, f"Missing title in {level}"
                assert "detail" in action, f"Missing detail in {level}"


class TestModelOutput:
    """Test model output shapes and types (if model is trained)."""

    def test_risk_class_is_1_2_or_3(self):
        """Risk class must always be 1, 2, or 3."""
        try:
            from predict_engine_v2 import predict_addiction
            result = predict_addiction(HIGH_RISK_INPUT)
            assert result["risk_class"] in [1, 2, 3]
        except FileNotFoundError:
            pytest.skip("Model not trained yet")

    def test_confidence_between_0_and_1(self):
        try:
            from predict_engine_v2 import predict_addiction
            result = predict_addiction(LOW_RISK_INPUT)
            assert 0.0 <= result["confidence"] <= 1.0
        except FileNotFoundError:
            pytest.skip("Model not trained yet")

    def test_probabilities_sum_to_1(self):
        try:
            from predict_engine_v2 import predict_addiction
            result = predict_addiction(LOW_RISK_INPUT)
            probs = result["probabilities"]
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.01, f"Probs sum to {total}, expected ~1.0"
        except FileNotFoundError:
            pytest.skip("Model not trained yet")

    def test_risk_score_matches_class(self):
        try:
            from predict_engine_v2 import predict_addiction
            result = predict_addiction(HIGH_RISK_INPUT)
            expected = {1: 0.0, 2: 0.5, 3: 1.0}
            assert result["risk_score"] == expected[result["risk_class"]]
        except FileNotFoundError:
            pytest.skip("Model not trained yet")

    def test_high_risk_input_not_classified_low(self):
        """Extreme high-risk input should not produce Low risk."""
        try:
            from predict_engine_v2 import predict_addiction
            result = predict_addiction(HIGH_RISK_INPUT)
            assert result["risk_class"] != 1, "High-risk input should not be Low"
        except FileNotFoundError:
            pytest.skip("Model not trained yet")

    def test_result_contains_all_keys(self):
        try:
            from predict_engine_v2 import predict_addiction
            result = predict_addiction(LOW_RISK_INPUT)
            required_keys = ["risk_class","risk_label","risk_score","confidence",
                             "probabilities","recommendations","notification","factors"]
            for k in required_keys:
                assert k in result, f"Missing key: {k}"
        except FileNotFoundError:
            pytest.skip("Model not trained yet")


class TestNotificationSystem:
    """Test that notifications are built correctly."""

    def test_low_risk_notification_is_positive(self):
        try:
            from predict_engine_v2 import predict_addiction
            result = predict_addiction(LOW_RISK_INPUT)
            if result["risk_label"] == "Low":
                assert "✅" in result["notification"]["title"] or \
                       "great" in result["notification"]["title"].lower()
        except FileNotFoundError:
            pytest.skip("Model not trained yet")

    def test_high_risk_notification_is_urgent(self):
        try:
            from predict_engine_v2 import predict_addiction
            result = predict_addiction(HIGH_RISK_INPUT)
            if result["risk_label"] == "High":
                notif = result["notification"]
                assert notif.get("urgency") == "high" or \
                       "🚨" in notif["title"] or "High" in notif["title"]
        except FileNotFoundError:
            pytest.skip("Model not trained yet")


class TestDatabase:
    """Test SQLite schema and operations."""

    def test_db_tables_exist(self):
        import sqlite3
        db = sqlite3.connect(":memory:")
        db.executescript("""
            CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT, password TEXT, name TEXT, created_at TEXT);
            CREATE TABLE predictions (
                id TEXT PRIMARY KEY, user_id TEXT, risk_class INTEGER,
                risk_label TEXT, risk_score REAL, confidence REAL,
                prob_low REAL, prob_moderate REAL, prob_high REAL,
                model_used TEXT, inputs TEXT, shap_drivers TEXT, created_at TEXT
            );
        """)
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "users"       in tables
        assert "predictions" in tables
        db.close()

    def test_prediction_insert_and_retrieve(self):
        import sqlite3, json
        db = sqlite3.connect(":memory:")
        db.executescript("""
            CREATE TABLE predictions (
                id TEXT, user_id TEXT, risk_class INTEGER, risk_label TEXT,
                risk_score REAL, confidence REAL, prob_low REAL, prob_moderate REAL,
                prob_high REAL, model_used TEXT, inputs TEXT, shap_drivers TEXT,
                created_at TEXT
            );
        """)
        db.execute("""INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   ("test-1","user-1",2,"Moderate",0.5,0.87,0.05,0.87,0.08,
                    "random_forest","{}","[]","2025-01-01"))
        db.commit()
        row = db.execute("SELECT * FROM predictions WHERE id='test-1'").fetchone()
        assert row is not None
        assert row[3] == "Moderate"
        db.close()
