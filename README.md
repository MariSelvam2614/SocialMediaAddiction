# 📱 SMA Predict v2.0 — Social Media Addiction Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Ensemble%20Stacking-orange)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-brightgreen)](https://shap.readthedocs.io/)
[![PWA](https://img.shields.io/badge/PWA-Ready-purple)](https://web.dev/progressive-web-apps/)

**SMA Predict v2.0** is an advanced, end-to-end Machine Learning solution designed to detect and analyze Social Media Addiction. It combines state-of-the-art predictive modeling with Explainable AI (SHAP) to provide users with deep insights into their behavioral patterns, personalized recommendations, and a premium web experience.

---

## 🚀 Key Features & Improvements

| Feature | Description |
| :--- | :--- |
| **🔥 Stacking Ensemble** | Combines Random Forest, Gradient Boosting, and XGBoost with a Logistic Regression meta-learner for superior accuracy. |
| **📉 SMOTE Integration** | Fixed substantial class imbalance issues, ensuring the model performs equally well on high-risk detection. |
| **🧠 Explainable AI (SHAP)** | Displays "Top Risk Drivers" for every single prediction, showing exactly which behaviors influenced the result. |
| **🎨 Premium UI/UX** | Modern 3-step wizard interface, live dashboards, and a cinematic user experience. |
| **📧 Smart Notifications** | Automated result emails via SMTP and scheduled 30-day reminders using APScheduler. |
| **📲 PWA Capabilities** | Installable on mobile/desktop with browser push notifications and offline support. |
| **🛡️ Secure Auth** | JWT-based authentication with Bcrypt password hashing and user history tracking. |
| **📊 Data Drift Detection** | Real-time monitoring of input distributions to alert when the model might need retraining. |

---

## 🛠️ Tech Stack

- **Backend:** Flask, SQLite, JWT (PyJWT), APScheduler
- **Machine Learning:** Scikit-learn, XGBoost, Optuna (Hyperparameter Tuning)
- **Explainability:** SHAP (Shapley Additive Explanations)
- **Frontend:** HTML5, CSS3, Vanilla JS (PWA enabled)
- **Testing:** Pytest (20+ test cases)

---

## 📂 Project Structure

```bash
Social-Media-Addiction/
├── data/               # Datasets and processed CSVs
├── models/             # Saved .pkl models and scalers
├── outputs/            # EDA reports, SHAP plots, and model comparisons
├── static/             # PWA assets (manifest, sw.js, icons)
├── templates/          # Jinja2 HTML templates
├── tests/              # Comprehensive Pytest suite
├── 01_data_fusion.py    # Data merging and feature engineering
├── 02_train_improved.py # Model training with Optuna tuning
├── 03_shap_explain.py   # Global and local explainability engine
├── 04_app_improved.py   # Main Flask Application
├── predict_engine_v2.py # Core ML inference logic
├── scheduler.py        # 30-day reminder service
└── requirements_core.txt # Python dependencies
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- Pip (Python Package Manager)

### 2. Install Dependencies
```bash
pip install -r requirements_core.txt
```

### 3. Initialize Environment
Generate the PWA icons and ensure directory structure is correct:
```bash
python push_notifications.py --generate-icons
```

---

## 🏃 Execution Guide

Follow these steps in order to train and run the system:

### Step 1: Data Pipeline
Fuse the datasets, perform feature engineering, and apply SMOTE.
```bash
python 01_data_fusion.py
```

### Step 2: Model Training
Train the ensemble stack with hyperparameter optimization.
```bash
python 02_train_improved.py
```

### Step 3: Explainability
Generate SHAP values and global summary plots.
```bash
python 03_shap_explain.py
```

### Step 4: Launch Application
Start the primary web server:
```bash
python 04_app_improved.py
```

Optional: Start the reminder scheduler in a separate terminal:
```bash
python scheduler.py
```

---

## 🧪 Testing
Run the automated test suite to verify model integrity and API endpoints:
```bash
pytest tests/ -v
```

---

## 📡 API Reference

| Endpoint | Method | Auth | Description |
| :--- | :--- | :--- | :--- |
| `/api/auth/register` | `POST` | None | User registration |
| `/api/auth/login` | `POST` | None | User login (returns JWT) |
| `/api/predict` | `POST` | JWT | Generate addiction prediction |
| `/api/history` | `GET` | JWT | Fetch assessment history |
| `/api/stats` | `GET` | None | Aggregate statistics |
| `/api/drift` | `GET` | JWT | Data drift report |

---

## 🛡️ Troubleshooting

- **Model not found?** Ensure you ran Steps 1 and 2 before starting the app.
- **Port 5000 busy?** Check if another Flask instance is running or use a different port in `04_app_improved.py`.
- **Emails not sending?** Verify your SMTP environment variables (`EMAIL_USER`, `EMAIL_PASS`).

---

Developed with ❤️ for Mental Health Awareness.
