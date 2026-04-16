"""
FILE: predict_engine_v2.py
IMPROVEMENTS: All improvements integrated — validation, SMOTE features,
              SHAP, notifications, history, engineered features
"""

import pandas as pd
import numpy as np
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import f1_score, classification_report
from imblearn.over_sampling  import SMOTE

MODEL_DIR  = "models"
DATA_DIR   = "data"
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_MAP   = {1: "Low", 2: "Moderate", 3: "High"}
LABEL_EMOJI = {1: "🟢", 2: "🟡", 3: "🔴"}

CORE_FEATURES = [
    "Gender_enc", "Living_Area", "Marital_Status",
    "Time_Spent", "Usage_Frequency", "Purpose_Use",
    "Neglect_Duties", "Preoccupation", "Withdrawal_Anxiety",
    "Attempted_Control", "Mood_Regulation", "Deception_Use",
    "Relationship_Impact"
]

ENGINEERED_FEATURES = CORE_FEATURES + [
    "Behavioral_Intensity", "Withdrawal_x_Neglect",
    "Control_Failure", "Social_Damage", "Craving_Score"
]

FIELD_RANGES = {
    "Gender_enc":          (0, 1),
    "Living_Area":         (1, 3),
    "Marital_Status":      (1, 3),
    "Time_Spent":          (1, 5),
    "Usage_Frequency":     (1, 5),
    "Purpose_Use":         (1, 5),
    "Neglect_Duties":      (1, 5),
    "Preoccupation":       (1, 5),
    "Withdrawal_Anxiety":  (1, 5),
    "Attempted_Control":   (1, 5),
    "Mood_Regulation":     (1, 5),
    "Deception_Use":       (1, 5),
    "Relationship_Impact": (1, 5),
}

RECOMMENDATIONS = {
    "Low": {
        "headline": "Your usage is in a healthy range — keep it up!",
        "summary":  "You show low signs of social media addiction. A few mindful habits will keep you on track.",
        "actions": [
            {"icon":"⏱️","title":"Set a weekly check-in",        "detail":"Review your screen-time stats once a week to stay aware of trends."},
            {"icon":"📵","title":"Phone-free meals",             "detail":"Keep your phone away during meals to build healthy offline boundaries."},
            {"icon":"🌿","title":"Nature breaks",                "detail":"Take a 10-min walk daily — no phone. This recharges focus."},
            {"icon":"😴","title":"Protect sleep quality",        "detail":"Avoid screens 30 min before bed to improve sleep depth."},
        ]
    },
    "Moderate": {
        "headline": "Moderate addiction detected — action recommended.",
        "summary":  "Your usage patterns show moderate dependency indicators.",
        "actions": [
            {"icon":"⏰","title":"App time limits",              "detail":"Set daily limits using your phone's Digital Wellbeing settings."},
            {"icon":"🚫","title":"App-free hours",               "detail":"Designate 3 hours each day as completely social-media-free."},
            {"icon":"🔕","title":"Disable notifications",        "detail":"Turn off all non-essential push notifications."},
            {"icon":"🛌","title":"No phone in bedroom",          "detail":"Charge your phone outside the bedroom. Use an alarm clock."},
            {"icon":"🧘","title":"Mindfulness practice",         "detail":"Try 5–10 min of mindfulness when you feel the urge to scroll."},
            {"icon":"👥","title":"Offline social activities",    "detail":"Schedule at least 2 in-person interactions per week."},
        ]
    },
    "High": {
        "headline": "High addiction risk — immediate changes needed.",
        "summary":  "Strong dependency patterns detected. Structured intervention is strongly recommended.",
        "actions": [
            {"icon":"🗑️","title":"Delete triggering apps",      "detail":"Temporarily remove the top 1–2 apps. Re-evaluate after 2 weeks."},
            {"icon":"🔒","title":"Use an app blocker",           "detail":"Install Freedom or Cold Turkey and schedule blocked hours daily."},
            {"icon":"📅","title":"Digital detox weekends",       "detail":"Commit to one full social-media-free weekend per month."},
            {"icon":"🤝","title":"Tell someone close",           "detail":"Share your goal with a trusted person who can hold you accountable."},
            {"icon":"🏃","title":"Replace with physical activity","detail":"Replace social media sessions with exercise — even 20 min/day."},
            {"icon":"📓","title":"Urge journaling",              "detail":"When compelled to scroll, write down the trigger instead."},
            {"icon":"🧠","title":"Seek professional support",    "detail":"Consider speaking with a counsellor specialising in digital addiction."},
            {"icon":"⏳","title":"Gradual reduction plan",       "detail":"Reduce usage by 30 min per week — gradual reduction is sustainable."},
        ]
    }
}

# ─── VALIDATION ───────────────────────────────────────────────────────────────
def validate_input(body: dict):
    errors = []
    cleaned = {}
    for field in CORE_FEATURES:
        if field not in body:
            errors.append(f"Missing required field: {field}")
            continue
        try:
            val = float(body[field])
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number")
            continue
        lo, hi = FIELD_RANGES.get(field, (0, 999))
        if not (lo <= val <= hi):
            errors.append(f"{field} must be between {lo} and {hi} (got {val})")
            continue
        cleaned[field] = val
    return errors, cleaned

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def engineer_features(clean: dict) -> dict:
    """Add 5 derived features to the cleaned input dict."""
    likert = ["Time_Spent","Usage_Frequency","Purpose_Use","Neglect_Duties",
              "Preoccupation","Withdrawal_Anxiety","Attempted_Control",
              "Mood_Regulation","Deception_Use","Relationship_Impact"]
    clean["Behavioral_Intensity"]  = float(np.mean([clean[k] for k in likert]))
    clean["Withdrawal_x_Neglect"]  = clean["Withdrawal_Anxiety"] * clean["Neglect_Duties"]
    clean["Control_Failure"]       = clean["Time_Spent"] * (6 - clean["Attempted_Control"])
    clean["Social_Damage"]         = clean["Relationship_Impact"] + clean["Deception_Use"]
    clean["Craving_Score"]         = clean["Preoccupation"] + clean["Withdrawal_Anxiety"]
    return clean

# ─── LAZY MODEL LOAD ──────────────────────────────────────────────────────────
_model, _scaler, _is_xgb, _shap_wrapper = None, None, False, None

def _load():
    global _model, _scaler, _is_xgb, _shap_wrapper
    if _model is not None:
        return
    if not os.path.exists(f"{MODEL_DIR}/best_model_name.txt"):
        raise FileNotFoundError("No trained model. Run 01_data_fusion.py and 02_train_improved.py first.")
    with open(f"{MODEL_DIR}/best_model_name.txt") as f:
        name = f.read().strip()
    _model  = joblib.load(f"{MODEL_DIR}/{name}.pkl")
    _scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    _is_xgb = "xgboost" in name.lower()
    try:
        _shap_wrapper = joblib.load(f"{MODEL_DIR}/shap_wrapper.pkl")
    except:
        pass

# ─── FACTOR ANALYSIS ─────────────────────────────────────────────────────────
def _analyse_factors(u: dict) -> list:
    factors = []
    if u.get("Time_Spent", 0) >= 4:
        factors.append({"name": "Excessive Time Spent",      "severity": "high",     "score": u["Time_Spent"]})
    if u.get("Withdrawal_Anxiety", 0) >= 4:
        factors.append({"name": "Withdrawal Anxiety",        "severity": "high",     "score": u["Withdrawal_Anxiety"]})
    if u.get("Neglect_Duties", 0) >= 4:
        factors.append({"name": "Neglecting Responsibilities","severity": "high",     "score": u["Neglect_Duties"]})
    if u.get("Mood_Regulation", 0) >= 4:
        factors.append({"name": "Mood Regulation via SMA",   "severity": "moderate", "score": u["Mood_Regulation"]})
    if u.get("Relationship_Impact", 0) >= 4:
        factors.append({"name": "Relationship Conflicts",    "severity": "moderate", "score": u["Relationship_Impact"]})
    if u.get("Preoccupation", 0) >= 4:
        factors.append({"name": "Constant Preoccupation",    "severity": "moderate", "score": u["Preoccupation"]})
    if u.get("Deception_Use", 0) >= 3:
        factors.append({"name": "Hiding Usage from Others",  "severity": "low",      "score": u["Deception_Use"]})
    if u.get("Attempted_Control", 0) <= 2:
        factors.append({"name": "Difficulty Controlling Use","severity": "moderate", "score": 6 - u.get("Attempted_Control",3)})
    return factors

# ─── PREDICT ─────────────────────────────────────────────────────────────────
def predict_addiction(input_dict: dict) -> dict:
    """
    Full prediction pipeline.
    Accepts raw user input (CORE_FEATURES), runs validation,
    feature engineering, inference, SHAP, and returns complete result.
    """
    _load()

    # Validate
    errors, cleaned = validate_input(input_dict)
    if errors:
        raise ValueError(f"Input validation failed: {errors}")

    # Engineer features
    full = engineer_features(cleaned.copy())

    # Scale
    row    = pd.DataFrame([[full[c] for c in ENGINEERED_FEATURES]], columns=ENGINEERED_FEATURES)
    row_s  = pd.DataFrame(_scaler.transform(row), columns=ENGINEERED_FEATURES)

    # Inference
    if _is_xgb:
        raw        = _model.predict(row_s)[0]
        proba      = _model.predict_proba(row_s)[0]
        risk_class = int(raw) + 1
    else:
        risk_class = int(_model.predict(row_s)[0])
        proba      = _model.predict_proba(row_s)[0]

    risk_label = LABEL_MAP[risk_class]
    confidence = float(max(proba))
    risk_score = round((risk_class - 1) / 2, 3)

    # SHAP
    shap_result = None
    if _shap_wrapper:
        try:
            shap_result = _shap_wrapper.explain(row_s)
        except:
            pass

    factors = _analyse_factors(cleaned)
    top     = factors[0]["name"] if factors else "Social Media Usage"
    rec     = RECOMMENDATIONS[risk_label]

    notification = {
        "Low":      {"title": "✅ You're doing great!",
                     "body":  f"Low risk detected. Top watch-point: {top}. Keep monitoring weekly.",
                     "color": "#22c982", "urgency": "low"},
        "Moderate": {"title": "⚠️ Moderate Addiction Detected",
                     "body":  f"{len(factors)} concern(s) flagged. Primary: {top}. Set daily usage limits now.",
                     "color": "#f5a623", "urgency": "medium"},
        "High":     {"title": "🚨 High Addiction Risk — Act Now",
                     "body":  f"{len(factors)} strong indicators. Primary: {top}. Immediate action recommended.",
                     "color": "#ef4444", "urgency": "high"},
    }[risk_label]

    return {
        "risk_class":      risk_class,
        "risk_label":      risk_label,
        "risk_emoji":      LABEL_EMOJI[risk_class],
        "risk_score":      risk_score,
        "confidence":      round(confidence, 4),
        "probabilities":   {
            "Low":      round(float(proba[0]), 4),
            "Moderate": round(float(proba[1]), 4),
            "High":     round(float(proba[2]), 4),
        },
        "factors":         factors,
        "shap_explanation":shap_result,
        "notification":    notification,
        "recommendations": rec,
    }


# ─── STANDALONE RUN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nTesting predict_engine_v2...\n")
    test_cases = [
        ("Low Risk User",  {
            "Gender_enc":1,"Living_Area":1,"Marital_Status":1,
            "Time_Spent":1,"Usage_Frequency":1,"Purpose_Use":5,
            "Neglect_Duties":1,"Preoccupation":1,"Withdrawal_Anxiety":1,
            "Attempted_Control":5,"Mood_Regulation":1,"Deception_Use":1,
            "Relationship_Impact":1
        }),
        ("High Risk User", {
            "Gender_enc":0,"Living_Area":2,"Marital_Status":2,
            "Time_Spent":5,"Usage_Frequency":5,"Purpose_Use":1,
            "Neglect_Duties":5,"Preoccupation":5,"Withdrawal_Anxiety":5,
            "Attempted_Control":1,"Mood_Regulation":5,"Deception_Use":4,
            "Relationship_Impact":5
        }),
    ]
    for label, inp in test_cases:
        try:
            r = predict_addiction(inp)
            print(f"  {label}: {r['risk_emoji']} {r['risk_label']} | "
                  f"Score={r['risk_score']} | Conf={r['confidence']:.0%}")
            print(f"    → {r['notification']['title']}")
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")
