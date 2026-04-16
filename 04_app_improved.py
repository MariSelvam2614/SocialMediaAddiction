"""
FILE: 04_app_improved.py
IMPROVEMENTS: SQLite persistence, JWT auth, email notifications,
              data drift detection, retraining pipeline, CORS, logging
Run: python 04_app_improved.py
"""

from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
import sqlite3, jwt, bcrypt, os, json, logging, warnings
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone, timedelta
from functools import wraps
from flask import render_template
import uuid
warnings.filterwarnings("ignore")

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder=".")
CORS(app)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
JWT_SECRET  = os.environ.get("JWT_SECRET",  "sma-jwt-secret-change-in-prod")
DB_PATH     = "data/predictions.db"
MODEL_DIR   = "models"
DATA_DIR    = "data"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
def load_model_artifacts():
    global _model, _scaler, _feature_cols, _best_name, _is_xgb, _shap_wrapper
    _model        = None
    _scaler       = None
    _feature_cols = None
    _best_name    = "unknown"
    _is_xgb       = False
    _shap_wrapper = None

    try:
        _feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
        _scaler       = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        with open(f"{MODEL_DIR}/best_model_name.txt") as f:
            _best_name = f.read().strip()
        _model  = joblib.load(f"{MODEL_DIR}/{_best_name}.pkl")
        _is_xgb = "xgboost" in _best_name.lower()
        logger.info(f"Model loaded: {_best_name}")
    except Exception as e:
        logger.warning(f"Model not found ({e}). Run 01_data_fusion.py + 02_train_improved.py first.")

    try:
        _shap_wrapper = joblib.load(f"{MODEL_DIR}/shap_wrapper.pkl")
        logger.info("SHAP explainer loaded")
    except:
        logger.info("SHAP explainer not available")

load_model_artifacts()

LABEL_MAP   = {1: "Low", 2: "Moderate", 3: "High"}
LABEL_EMOJI = {1: "🟢", 2: "🟡", 3: "🔴"}

# ─── DATABASE ─────────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db: db.close()

def init_db():
    with app.app_context():
        db = sqlite3.connect(DB_PATH)
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                email       TEXT UNIQUE NOT NULL,
                password    TEXT NOT NULL,
                name        TEXT,
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id              TEXT PRIMARY KEY,
                user_id         TEXT,
                risk_class      INTEGER,
                risk_label      TEXT,
                risk_score      REAL,
                confidence      REAL,
                prob_low        REAL,
                prob_moderate   REAL,
                prob_high       REAL,
                model_used      TEXT,
                inputs          TEXT,
                shap_drivers    TEXT,
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS drift_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                feature     TEXT,
                train_mean  REAL,
                live_mean   REAL,
                drift_pct   REAL,
                flagged     INTEGER DEFAULT 0,
                logged_at   TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        db.commit()
        db.close()
    logger.info("✓ Database initialised")

init_db()

# ─── JWT AUTH ─────────────────────────────────────────────────────────────────
def create_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Missing token"}), 401
        try:
            token   = auth.split(" ")[1]
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            g.user_id = payload["sub"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return wrapper

# ─── INFERENCE ────────────────────────────────────────────────────────────────
def run_inference(input_dict: dict):
    if _model is None:
        raise RuntimeError("No model loaded. Run training scripts first.")

    row = pd.DataFrame([[input_dict[c] for c in _feature_cols]], columns=_feature_cols)
    row_s = pd.DataFrame(_scaler.transform(row), columns=_feature_cols)

    if _is_xgb:
        raw   = _model.predict(row_s)[0]
        proba = _model.predict_proba(row_s)[0]
        risk_class = int(raw) + 1
    else:
        risk_class = int(_model.predict(row_s)[0])
        proba      = _model.predict_proba(row_s)[0]

    # SHAP explanation
    shap_result = None
    if _shap_wrapper:
        try:
            shap_result = _shap_wrapper.explain(row_s)
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

    return risk_class, proba.tolist(), row_s, shap_result

# ─── EMAIL NOTIFICATION ───────────────────────────────────────────────────────
def send_email_notification(to_email: str, user_name: str, result: dict):
    """
    Send prediction result via email.
    Configure SMTP credentials in environment variables:
      EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText

    host = os.environ.get("EMAIL_HOST")
    port = int(os.environ.get("EMAIL_PORT", 587))
    user = os.environ.get("EMAIL_USER")
    pwd  = os.environ.get("EMAIL_PASS")

    if not all([host, user, pwd]):
        logger.info("Email not configured — skipping notification")
        return False

    rl    = result["risk_label"]
    emoji = result["risk_emoji"]
    recs  = result["recommendations"]["actions"][:3]
    rec_html = "".join(f"<li><b>{r['icon']} {r['title']}</b> — {r['detail']}</li>" for r in recs)

    color = {"Low": "#22c982", "Moderate": "#f5a623", "High": "#ef4444"}[rl]

    html = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:0 auto;background:#0d1117;color:#dde6f0;border-radius:12px;overflow:hidden">
      <div style="background:{color};padding:24px;text-align:center">
        <h1 style="color:#fff;margin:0;font-size:1.4rem">{emoji} Your Addiction Risk Result</h1>
      </div>
      <div style="padding:28px">
        <p>Hi {user_name},</p>
        <p>Your SMA Predict assessment is complete. Here is your result:</p>
        <div style="background:#161f2e;border-radius:8px;padding:20px;margin:20px 0;border-left:4px solid {color}">
          <h2 style="margin:0 0 8px;color:{color}">{rl} Risk</h2>
          <p style="margin:0;color:#8aaac8">
            Risk Score: {result['risk_score']} &nbsp;|&nbsp;
            Confidence: {round(result['confidence']*100)}%
          </p>
        </div>
        <h3 style="color:#fff">Your Top 3 Recommendations</h3>
        <ul style="color:#9bb4cc;line-height:1.8">{rec_html}</ul>
        <p style="color:#56728a;font-size:0.8rem;margin-top:28px">
          ⚕️ This is an awareness tool only and does not replace professional advice.
        </p>
      </div>
    </div>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"SMA Predict — Your Risk Assessment: {rl} Risk {emoji}"
    msg["From"]    = user
    msg["To"]      = to_email
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, pwd)
            server.sendmail(user, to_email, msg.as_string())
        logger.info(f"Email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False

# ─── DATA DRIFT DETECTION ─────────────────────────────────────────────────────
def check_data_drift(new_input_df: pd.DataFrame, threshold: float = 0.25) -> list:
    """Compare new input means against training data means. Flag >25% drift."""
    drift_flags = []
    try:
        X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
        for col in _feature_cols:
            if col not in X_train.columns:
                continue
            train_mean = X_train[col].mean()
            live_mean  = new_input_df[col].mean()
            if train_mean == 0:
                continue
            drift_pct = abs(live_mean - train_mean) / (abs(train_mean) + 1e-9)
            flagged   = drift_pct > threshold

            db = sqlite3.connect(DB_PATH)
            db.execute("""INSERT INTO drift_log
                          (feature, train_mean, live_mean, drift_pct, flagged)
                          VALUES (?,?,?,?,?)""",
                       (col, round(train_mean,4), round(live_mean,4),
                        round(drift_pct,4), int(flagged)))
            db.commit(); db.close()

            if flagged:
                drift_flags.append({"feature": col, "drift_pct": round(drift_pct,3)})
    except Exception as e:
        logger.warning(f"Drift check failed: {e}")

    return drift_flags

# ─── RECOMMENDATIONS ─────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Low": {
        "headline": "Your usage is in a healthy range — keep it up!",
        "summary":  "You show low signs of social media addiction.",
        "actions": [
            {"icon":"⏱️","title":"Set a weekly check-in",       "detail":"Review your screen-time stats once a week to stay aware of trends."},
            {"icon":"📵","title":"Phone-free meals",             "detail":"Keep your phone away during meals to build healthy offline boundaries."},
            {"icon":"🌿","title":"Nature breaks",                "detail":"Take a 10-min walk daily — no phone. This recharges focus."},
            {"icon":"😴","title":"Protect sleep quality",        "detail":"Avoid screens 30 min before bed to improve sleep depth."},
        ]
    },
    "Moderate": {
        "headline": "Moderate addiction detected — action recommended.",
        "summary":  "Your usage shows moderate dependency indicators.",
        "actions": [
            {"icon":"⏰","title":"App time limits",              "detail":"Set daily limits using your phone's Digital Wellbeing / Screen Time settings."},
            {"icon":"🚫","title":"App-free hours",               "detail":"Designate 3 hours each day as completely social-media-free."},
            {"icon":"🔕","title":"Disable notifications",        "detail":"Turn off all non-essential push notifications."},
            {"icon":"🛌","title":"No phone in bedroom",          "detail":"Charge your phone outside the bedroom. Use an alarm clock."},
            {"icon":"🧘","title":"Mindfulness practice",         "detail":"Try 5–10 min of mindfulness when you feel the urge to scroll."},
            {"icon":"👥","title":"Offline social activities",    "detail":"Schedule at least 2 in-person interactions per week."},
        ]
    },
    "High": {
        "headline": "High addiction risk — immediate changes needed.",
        "summary":  "Strong dependency patterns detected.",
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

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")
    
# DASHBOARD PAGE (ADDED)
@app.route("/dashboard")
def dashboard():
    return send_from_directory(".", "dashboard_live.html")

# AUTH
@app.route("/api/auth/register", methods=["POST"])
def register():
    body = request.get_json(force=True, silent=True) or {}
    email = body.get("email","").strip().lower()
    name  = body.get("name","").strip()
    pwd   = body.get("password","")

    if not email or not pwd:
        return jsonify({"error": "email and password required"}), 400
    if len(pwd) < 6:
        return jsonify({"error": "password must be ≥6 characters"}), 400

    db = get_db()
    if db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
        return jsonify({"error": "Email already registered"}), 409

    user_id  = str(uuid.uuid4())
    hashed   = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
    db.execute("INSERT INTO users (id,email,password,name) VALUES (?,?,?,?)",
               (user_id, email, hashed, name))
    db.commit()
    token = create_token(user_id)
    logger.info(f"New user registered: {email}")
    return jsonify({"token": token, "user_id": user_id, "name": name}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    body  = request.get_json(force=True, silent=True) or {}
    email = body.get("email","").strip().lower()
    pwd   = body.get("password","")

    db   = get_db()
    user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not user or not bcrypt.checkpw(pwd.encode(), user["password"].encode()):
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_token(user["id"])
    return jsonify({"token": token, "user_id": user["id"], "name": user["name"]}), 200

# PREDICT
@app.route("/api/predict", methods=["POST"])
@require_auth
def predict():
    body = request.get_json(force=True, silent=True) or {}

    required = [
        "Gender_enc","Living_Area","Marital_Status",
        "Time_Spent","Usage_Frequency","Purpose_Use",
        "Neglect_Duties","Preoccupation","Withdrawal_Anxiety",
        "Attempted_Control","Mood_Regulation","Deception_Use",
        "Relationship_Impact"
    ]
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 400

    try:
        clean = {k: float(body[k]) for k in required}

        # Engineered features
        likert = ["Time_Spent","Usage_Frequency","Purpose_Use","Neglect_Duties",
                  "Preoccupation","Withdrawal_Anxiety","Attempted_Control",
                  "Mood_Regulation","Deception_Use","Relationship_Impact"]
        clean["Behavioral_Intensity"]  = np.mean([clean[k] for k in likert])
        clean["Withdrawal_x_Neglect"]  = clean["Withdrawal_Anxiety"] * clean["Neglect_Duties"]
        clean["Control_Failure"]       = clean["Time_Spent"] * (6 - clean["Attempted_Control"])
        clean["Social_Damage"]         = clean["Relationship_Impact"] + clean["Deception_Use"]
        clean["Craving_Score"]         = clean["Preoccupation"] + clean["Withdrawal_Anxiety"]

        risk_class, proba, row_scaled, shap_result = run_inference(clean)

        # Drift check
        drift_flags = check_data_drift(row_scaled)
        if drift_flags:
            logger.warning(f"Data drift detected: {drift_flags}")

        risk_label = LABEL_MAP[risk_class]
        confidence = float(max(proba))
        rec        = RECOMMENDATIONS[risk_label]

        result = {
            "prediction_id":   str(uuid.uuid4())[:8],
            "risk_class":      risk_class,
            "risk_label":      risk_label,
            "risk_emoji":      LABEL_EMOJI[risk_class],
            "risk_score":      round((risk_class - 1) / 2, 3),
            "confidence":      round(confidence, 4),
            "probabilities":   {
                "Low":      round(proba[0], 4),
                "Moderate": round(proba[1], 4),
                "High":     round(proba[2], 4),
            },
            "shap_explanation": shap_result,
            "drift_warnings":  drift_flags,
            "recommendations": rec,
            "notification": {
                "Low":      {"title":"✅ You're doing great!",             "body": f"Low risk. Top recommendation: {rec['actions'][0]['title']}."},
                "Moderate": {"title":"⚠️ Moderate Addiction Detected",     "body": f"{len(drift_flags)+2} concern(s) flagged. Start with: {rec['actions'][0]['title']}."},
                "High":     {"title":"🚨 High Addiction Risk — Act Now",   "body": f"Strong dependency indicators. Immediate action: {rec['actions'][0]['title']}."},
            }[risk_label],
            "model_used": _best_name,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }

        # Persist to DB
        db = get_db()
        db.execute("""INSERT INTO predictions
            (id,user_id,risk_class,risk_label,risk_score,confidence,
             prob_low,prob_moderate,prob_high,model_used,inputs,shap_drivers,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result["prediction_id"], g.user_id, risk_class, risk_label,
             result["risk_score"], confidence,
             proba[0], proba[1], proba[2], _best_name,
             json.dumps({k: body[k] for k in required}),
             json.dumps(shap_result.get("top_drivers",[]) if shap_result else []),
             result["timestamp"]))
        db.commit()
        logger.info(f"Prediction {result['prediction_id']}: {risk_label} ({g.user_id})")

        # Optional email
        if body.get("send_email"):
            user = get_db().execute("SELECT * FROM users WHERE id=?", (g.user_id,)).fetchone()
            if user:
                send_email_notification(user["email"], user["name"] or "User", result)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Predict error: {e}")
        return jsonify({"error": str(e)}), 500

# HISTORY
@app.route("/api/history", methods=["GET"])
@require_auth
def history():
    limit  = min(int(request.args.get("limit",  20)), 100)
    offset = int(request.args.get("offset", 0))
    db     = get_db()

    rows = db.execute("""
        SELECT id, risk_label, risk_score, confidence, model_used, created_at
        FROM predictions WHERE user_id=?
        ORDER BY created_at DESC LIMIT ? OFFSET ?
    """, (g.user_id, limit, offset)).fetchall()

    total = db.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?",
                       (g.user_id,)).fetchone()[0]

    return jsonify({
        "total": total,
        "predictions": [dict(r) for r in rows]
    }), 200

# STATS (aggregate)
@app.route("/api/stats", methods=["GET"])
def stats():
    db = get_db()
    total = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if total == 0:
        return jsonify({"total": 0})

    dist  = db.execute("""
        SELECT risk_label, COUNT(*) as cnt
        FROM predictions GROUP BY risk_label
    """).fetchall()
    avg_conf = db.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
    avg_score= db.execute("SELECT AVG(risk_score) FROM predictions").fetchone()[0]

    return jsonify({
        "total_predictions": total,
        "risk_distribution": {r["risk_label"]: r["cnt"] for r in dist},
        "avg_confidence":    round(avg_conf or 0, 4),
        "avg_risk_score":    round(avg_score or 0, 4),
    }), 200

# DRIFT REPORT
@app.route("/api/drift", methods=["GET"])
@require_auth
def drift_report():
    db   = get_db()
    rows = db.execute("""
        SELECT feature, AVG(drift_pct) as avg_drift, MAX(flagged) as ever_flagged
        FROM drift_log GROUP BY feature ORDER BY avg_drift DESC
    """).fetchall()
    return jsonify({"drift": [dict(r) for r in rows]}), 200

# RETRAIN
@app.route("/api/retrain", methods=["POST"])
@require_auth
def retrain():
    try:
        import subprocess
        subprocess.Popen(["python", "01_data_fusion.py"])
        subprocess.Popen(["python", "02_train_improved.py"])
        return jsonify({"message": "Retraining pipeline started"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# HEALTH
@app.route("/api/health")
def health():
    return jsonify({
        "status":      "ok",
        "model":       _best_name,
        "model_ready": _model is not None,
        "shap_ready":  _shap_wrapper is not None,
        "version":     "2.0.0"
    }), 200

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  SMA Predict v2.0 — http://localhost:5000")
    print("="*50)
    app.run(debug=True, port=5000)
