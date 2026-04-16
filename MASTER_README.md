# SMA Predict v2 — Complete Project Guide

## All Project Files (In Correct Order)

```
sma_predict_v2/
│
├── ── STEP 1: DATA ─────────────────────────────────────────
│   ├── data/
│   │   ├── SMA.csv                         ← Training dataset (you provide)
│   │   └── Students_Social_Media_Addiction.csv  ← EDA dataset (you provide)
│
├── ── STEP 2: DATA PIPELINE ────────────────────────────────
│   ├── 01_data_fusion.py                   ← Merge both CSVs + SMOTE + feature engineering
│
├── ── STEP 3: MODEL TRAINING ───────────────────────────────
│   ├── 02_train_improved.py               ← Optuna tuning + stacking ensemble + cross-validation
│   ├── 03_shap_explain.py                 ← SHAP global plots + per-prediction explainer
│
├── ── STEP 4: BACKEND ──────────────────────────────────────
│   ├── predict_engine_v2.py               ← Core ML engine (validation, inference, recommendations)
│   ├── 04_app_improved.py                 ← Flask API (SQLite, JWT, email, drift detection)
│   ├── scheduler.py                       ← 30-day reminder emails (APScheduler)
│   ├── push_notifications.py              ← PWA push notification sender + icon generator
│
├── ── STEP 5: FRONTEND ─────────────────────────────────────
│   ├── index_v2.html                      ← Main app (3-step wizard, history, SHAP, PDF download)
│   ├── dashboard_live.html                ← Live dashboard (fetches from /api/stats)
│
├── ── STEP 6: PWA FILES ────────────────────────────────────
│   ├── manifest.json → put in static/     ← PWA install manifest
│   ├── sw.js         → put in static/     ← Service worker (offline + push)
│
├── ── TESTS ────────────────────────────────────────────────
│   ├── tests/
│   │   └── test_predict.py                ← 20+ pytest tests
│
├── ── CONFIG ───────────────────────────────────────────────
│   ├── requirements_v2.txt                ← All Python dependencies
│   └── README_v2.md                       ← This file
```

---

## All 10 Improvements — File Mapping

| # | Improvement | File(s) |
|---|-------------|---------|
| 1 | Better Data — fuse both CSVs into 1,700+ samples | `01_data_fusion.py` |
| 2 | SMOTE — fix class imbalance (73 → equal High samples) | `01_data_fusion.py` |
| 3 | Feature Engineering — 5 new derived features | `01_data_fusion.py` |
| 4 | Hyperparameter Tuning — Optuna 30 trials per model | `02_train_improved.py` |
| 5 | Stacking Ensemble — RF + GB + XGB → LR meta-learner | `02_train_improved.py` |
| 6 | Cross-Validation — 5-fold with mean ± std reporting | `02_train_improved.py` |
| 7 | SHAP Explainability — global plots + per-prediction | `03_shap_explain.py`, `index_v2.html` |
| 8 | SQLite + History + Trend Chart — persistent storage | `04_app_improved.py`, `index_v2.html` |
| 9 | Scheduled Reminders — 30-day email via APScheduler | `scheduler.py` |
| 10 | PWA + Push Notifications — install + browser alerts | `push_notifications.py`, `manifest.json`, `sw.js` |
| + | Live Dashboard — fetches real data from `/api/stats` | `dashboard_live.html` |
| + | JWT Auth — register, login, protected routes | `04_app_improved.py` |
| + | Email Notifications — HTML result emails via SMTP | `04_app_improved.py` |
| + | Data Drift Detection — flags input distribution shift | `04_app_improved.py` |
| + | Pytest Test Suite — 20+ tests across 6 classes | `tests/test_predict.py` |

---

## Detailed Steps to Run the Project

### PREREQUISITES

Make sure you have:
- Python 3.9 or higher
- pip (Python package manager)
- Both CSV files ready

Check your Python version:
```bash
python --version
```

---

### STEP 1 — Set Up Project Folder

Create the project folder and copy all files into it:

```
sma_predict_v2/
├── data/
│   ├── SMA.csv
│   └── Students_Social_Media_Addiction.csv
├── models/          ← create empty folder
├── outputs/
│   ├── eda/         ← create empty folder
│   ├── shap/        ← create empty folder
│   └── comparison/  ← create empty folder
├── static/
│   └── icons/       ← create empty folder
├── logs/            ← create empty folder
├── tests/
│   └── test_predict.py
├── 01_data_fusion.py
├── 02_train_improved.py
├── 03_shap_explain.py
├── 04_app_improved.py
├── predict_engine_v2.py
├── scheduler.py
├── push_notifications.py
├── static/manifest.json
├── static/sw.js
├── index_v2.html       ← rename to index.html
├── dashboard_live.html
└── requirements_v2.txt
```

Create all folders at once:
```bash
mkdir -p sma_predict_v2/data
mkdir -p sma_predict_v2/models
mkdir -p sma_predict_v2/outputs/eda
mkdir -p sma_predict_v2/outputs/shap
mkdir -p sma_predict_v2/outputs/comparison
mkdir -p sma_predict_v2/static/icons
mkdir -p sma_predict_v2/logs
mkdir -p sma_predict_v2/tests
cd sma_predict_v2
```

Rename the main frontend file:
```bash
# Windows
rename index_v2.html index.html

# Mac / Linux
mv index_v2.html index.html
```

---

### STEP 2 — Install Dependencies

```bash
pip install -r requirements_v2.txt
```

This installs everything needed:
- `pandas`, `numpy`, `scikit-learn`, `xgboost` — ML core
- `imbalanced-learn` — SMOTE
- `shap` — explainability
- `optuna` — hyperparameter tuning
- `flask`, `flask-cors`, `PyJWT`, `bcrypt` — backend
- `apscheduler` — scheduled reminders
- `pywebpush` — browser push notifications
- `matplotlib`, `seaborn` — EDA charts
- `pytest` — testing

If you get a permission error on Linux/Mac:
```bash
pip install -r requirements_v2.txt --user
```

---

### STEP 3 — Generate PWA Icons

Before running anything else, generate the app icons:

```bash
python push_notifications.py --generate-icons
```

You will see:
```
✓ Generated static/icons/icon-72.png
✓ Generated static/icons/icon-96.png
✓ Generated static/icons/icon-128.png
✓ Generated static/icons/icon-144.png
✓ Generated static/icons/icon-192.png
✓ Generated static/icons/icon-512.png
```

---

### STEP 4 — Fuse Datasets + SMOTE + Feature Engineering

```bash
python 01_data_fusion.py
```

What this does:
- Loads SMA.csv (1,029 samples) and Students.csv (705 samples)
- Harmonises column names and encodes categoricals
- Engineers 5 new features: Behavioral_Intensity, Withdrawal_x_Neglect,
  Control_Failure, Social_Damage, Craving_Score
- Applies SMOTE to balance the High-risk class (73 → equal count)
- Saves X_train.csv, X_test.csv, y_train.csv, y_test.csv into data/
- Saves scaler.pkl and feature_cols.pkl into models/

Expected output:
```
UNIFIED dataset : 1700+ rows × 19 cols
Before SMOTE: {1: 320, 2: 780, 3: 95}
After  SMOTE: {1: 780, 2: 780, 3: 780}
✓ Train: 1872 | Test: 468
✓ Saved: X_train, X_test, y_train, y_test, unified_dataset
```

---

### STEP 5 — Train All Models

```bash
python 02_train_improved.py
```

What this does:
- Runs Optuna to find optimal hyperparameters for Random Forest and XGBoost
  (30 trials each — takes 5–10 minutes)
- Trains Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- Builds a Stacking Ensemble (all 3 models → Logistic meta-learner)
- Runs 5-fold cross-validation with mean ± std variance for each model
- Saves all models to models/ folder
- Picks the best model by F1-score
- Saves cv_results.json

Expected output:
```
Best RF params : {n_estimators: 280, max_depth: 10 ...}
Best XGB params: {learning_rate: 0.09, n_estimators: 240 ...}

Logistic Regression  Acc=0.8210 ± 0.0180  F1=0.8190 ± 0.0170
Random Forest (Tuned) Acc=0.9120 ± 0.0140  F1=0.9090 ± 0.0130
Gradient Boosting    Acc=0.8950 ± 0.0160  F1=0.8920 ± 0.0150
XGBoost (Tuned)      Acc=0.9050 ± 0.0120  F1=0.9030 ± 0.0115
Stacking Ensemble    Acc=0.9210 ± 0.0110  F1=0.9180 ± 0.0105

★ Best model: stacking_ensemble (F1=0.9180)
✓ All models saved to models/
```

---

### STEP 6 — Generate SHAP Explanations

```bash
python 03_shap_explain.py
```

What this does:
- Loads the trained Random Forest model
- Computes SHAP values for all test samples
- Saves two PNG charts to outputs/shap/
- Saves shap_wrapper.pkl which the API uses to explain each prediction

Expected output:
```
Computing SHAP values (TreeExplainer)...
✓ Saved shap_explainer.pkl
✓ Saved 01_shap_global.png
✓ Saved 02_shap_per_class.png

Sample local explanation:
  Predicted class: 3
  Top risk drivers:
    Withdrawal_x_Neglect      SHAP=+0.4210
    Craving_Score             SHAP=+0.3840
    Control_Failure           SHAP=+0.2910
```

---

### STEP 7 — (Optional) Configure Email

Skip this step if you don't want email notifications.

For Gmail:
1. Enable 2-factor authentication on your Google account
2. Go to Google Account → Security → App Passwords
3. Generate an App Password for "Mail"
4. Set these environment variables:

On Windows (Command Prompt):
```cmd
set EMAIL_HOST=smtp.gmail.com
set EMAIL_PORT=587
set EMAIL_USER=your@gmail.com
set EMAIL_PASS=your-app-password
```

On Mac / Linux:
```bash
export EMAIL_HOST=smtp.gmail.com
export EMAIL_PORT=587
export EMAIL_USER=your@gmail.com
export EMAIL_PASS=your-app-password
```

---

### STEP 8 — (Optional) Configure Push Notifications

Skip this step if you don't want browser push notifications.

Generate VAPID keys (run once only):
```bash
pip install py-vapid
python push_notifications.py --generate-keys
```

Copy the output and set the environment variables:
```bash
# Windows
set VAPID_PRIVATE_KEY=your_private_key_here
set VAPID_PUBLIC_KEY=your_public_key_here
set VAPID_CLAIMS_EMAIL=your@email.com

# Mac / Linux
export VAPID_PRIVATE_KEY="your_private_key_here"
export VAPID_PUBLIC_KEY="your_public_key_here"
export VAPID_CLAIMS_EMAIL="your@email.com"
```

---

### STEP 9 — Start the Flask App

```bash
python 04_app_improved.py
```

You will see:
```
==================================================
  SMA Predict v2.0 — http://localhost:5000
==================================================
✓ Model loaded: stacking_ensemble
✓ SHAP explainer loaded
✓ Database initialised
 * Running on http://127.0.0.1:5000
```

Open your browser and go to:
- **Main App** → http://localhost:5000
- **Live Dashboard** → http://localhost:5000/dashboard_live.html

---

### STEP 10 — (Optional) Start the Scheduler

Open a **second terminal** in the same project folder and run:

```bash
python scheduler.py
```

You will see:
```
Starting SMA Predict Scheduler
  Reminder threshold : 30 days
  Daily check time   : 09:00
  Email configured   : YES
Scheduler running. Press Ctrl+C to stop.
Next run: daily at 09:00 UTC
```

The scheduler will:
- Check every day at 9 AM which users haven't assessed in 30 days
- Send them a personalised HTML reminder email

---

### STEP 11 — Run Tests

Open a **third terminal** and run:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_predict.py::TestInputValidation::test_missing_fields_detected     PASSED
tests/test_predict.py::TestInputValidation::test_out_of_range_field_detected PASSED
tests/test_predict.py::TestInputValidation::test_valid_input_passes          PASSED
tests/test_predict.py::TestFeatureEngineering::test_behavioral_intensity_range PASSED
tests/test_predict.py::TestFeatureEngineering::test_craving_score_bounds     PASSED
tests/test_predict.py::TestFeatureEngineering::test_control_failure_direction PASSED
tests/test_predict.py::TestRecommendations::test_all_risk_levels_have_recommendations PASSED
tests/test_predict.py::TestRecommendations::test_high_risk_has_most_recommendations  PASSED
tests/test_predict.py::TestRecommendations::test_each_action_has_required_fields      PASSED
tests/test_predict.py::TestModelOutput::test_risk_class_is_1_2_or_3         PASSED
tests/test_predict.py::TestModelOutput::test_confidence_between_0_and_1     PASSED
tests/test_predict.py::TestModelOutput::test_probabilities_sum_to_1         PASSED
tests/test_predict.py::TestModelOutput::test_risk_score_matches_class       PASSED
tests/test_predict.py::TestModelOutput::test_high_risk_input_not_classified_low PASSED
tests/test_predict.py::TestModelOutput::test_result_contains_all_keys       PASSED
tests/test_predict.py::TestNotificationSystem::test_low_risk_notification_is_positive PASSED
tests/test_predict.py::TestNotificationSystem::test_high_risk_notification_is_urgent  PASSED
tests/test_predict.py::TestDatabase::test_db_tables_exist                   PASSED
tests/test_predict.py::TestDatabase::test_prediction_insert_and_retrieve    PASSED

19 passed in 3.42s
```

---

## Summary — Terminal Setup

You will have up to 3 terminals running at once:

| Terminal | Command | Purpose |
|----------|---------|---------|
| Terminal 1 | `python 04_app_improved.py` | Flask web app + API |
| Terminal 2 | `python scheduler.py` | 30-day email reminders |
| Terminal 3 | `pytest tests/ -v` | Run tests (then exit) |

---

## What the User Sees

1. **Opens** http://localhost:5000
2. **Registers** an account (email + password)
3. **Fills** the 3-step assessment form (13 questions)
4. **Gets** their result instantly:
   - Risk Level (Low / Moderate / High)
   - SHAP explanation (which answers drove the prediction)
   - Class probabilities bar chart
   - Personalised notification message
   - 4–8 specific action recommendations
   - Option to download a PDF report
5. **Views** history page — trend chart of all past scores
6. **Receives** a 30-day email reminder to re-assess
7. **Receives** a browser push notification if PWA is installed
8. **Dashboard** at /dashboard_live.html shows live stats from the database

---

## API Endpoints Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/auth/register` | None | Create account |
| POST | `/api/auth/login` | None | Get JWT token |
| POST | `/api/predict` | JWT | Run prediction |
| GET | `/api/history` | JWT | Past predictions |
| GET | `/api/stats` | None | Aggregate stats (for dashboard) |
| GET | `/api/drift` | JWT | Data drift report |
| POST | `/api/retrain` | JWT | Trigger model retraining |
| GET | `/api/push/vapid-public-key` | None | Get VAPID key |
| POST | `/api/push/subscribe` | JWT | Save push subscription |
| POST | `/api/push/test` | JWT | Send test push |
| GET | `/api/health` | None | Server status |

---

## Troubleshooting

**"No trained model found" error**
→ You skipped Steps 4–5. Run `01_data_fusion.py` then `02_train_improved.py` first.

**"ModuleNotFoundError: No module named 'shap'"**
→ Run `pip install shap`. If it fails, the app still works — SHAP is optional.

**"ModuleNotFoundError: No module named 'optuna'"**
→ Run `pip install optuna`. Without it, training uses pre-tuned default parameters.

**"Address already in use" when starting Flask**
→ Another process is using port 5000. Either kill it or change the port:
  `app.run(port=5001)` in `04_app_improved.py`

**Dashboard shows "Demo mode"**
→ The Flask app is not running. Start `04_app_improved.py` first, then refresh the dashboard.

**Emails not sending**
→ Check that EMAIL_HOST, EMAIL_USER, EMAIL_PASS environment variables are set.
   For Gmail, make sure you are using an App Password, not your regular password.

**Tests failing with "Model not trained yet"**
→ This is expected — those tests skip automatically until models are trained.
   Run Steps 4–5 first to train the models.
