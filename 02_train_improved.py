"""
FILE: 02_train_improved.py
IMPROVEMENTS: Hyperparameter Tuning + Stacking Ensemble + Cross-Validation
"""

import os, json, warnings, joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
DATA_DIR  = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------
print("=" * 60)
print("STEP 1 — Loading Preprocessed Data")
print("=" * 60)

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Classes: {sorted(y_train.unique())}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------------------
def cv_report(name, model, X, y):
    res = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "f1_weighted"],
        return_train_score=True,
        n_jobs=-1
    )
    print(f"\n{name}")
    print(f"  CV Acc : {res['test_accuracy'].mean():.4f} ± {res['test_accuracy'].std():.4f}")
    print(f"  CV F1  : {res['test_f1_weighted'].mean():.4f} ± {res['test_f1_weighted'].std():.4f}")
    print(f"  Train F1: {res['train_f1_weighted'].mean():.4f}")

    return {
        "name": name,
        "cv_acc_mean": round(res["test_accuracy"].mean(), 4),
        "cv_acc_std":  round(res["test_accuracy"].std(), 4),
        "cv_f1_mean":  round(res["test_f1_weighted"].mean(), 4),
        "cv_f1_std":   round(res["test_f1_weighted"].std(), 4),
    }

# ---------------------------------------------------------------------
print("\nSTEP 2 — Hyperparameter Tuning (Optuna)")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5])
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        return cross_validate(
            model, X_train, y_train,
            cv=cv, scoring="f1_weighted", n_jobs=-1
        )["test_score"].mean()

    rf_study = optuna.create_study(direction="maximize")
    rf_study.optimize(rf_objective, n_trials=30)
    best_rf_params = rf_study.best_params

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = XGBClassifier(
            **params,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0
        )
        return cross_validate(
            model, X_train, y_train - 1,
            cv=cv, scoring="f1_weighted", n_jobs=-1
        )["test_score"].mean()

    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(xgb_objective, n_trials=30)
    best_xgb_params = xgb_study.best_params
    TUNED = True

except ImportError:
    best_rf_params  = {"n_estimators": 250, "max_depth": 12, "min_samples_leaf": 2, "max_features": "sqrt"}
    best_xgb_params = {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.08, "subsample": 0.85, "colsample_bytree": 0.8}
    TUNED = False

# ---------------------------------------------------------------------
print("\nSTEP 3 — Cross-Validation Report")

rf_model = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
lr_model = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)

xgb_model = XGBClassifier(
    **best_xgb_params,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0
)

cv_results = []
cv_results.append(cv_report("Logistic Regression", lr_model, X_train, y_train))
cv_results.append(cv_report("Random Forest", rf_model, X_train, y_train))
cv_results.append(cv_report("Gradient Boosting", gb_model, X_train, y_train))
cv_results.append(cv_report("XGBoost", xgb_model, X_train, y_train - 1))

# ---------------------------------------------------------------------
print("\nSTEP 4 — Stacking Ensemble")

class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.verbosity = verbosity

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric=self.eval_metric,
            verbosity=self.verbosity
        )

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

estimators = [
    ("rf", rf_model),
    ("gb", gb_model),
    ("xgb", XGBWrapper(**best_xgb_params))
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, multi_class="multinomial"),
    cv=5,
    n_jobs=-1
)

cv_results.append(cv_report("Stacking Ensemble", stack, X_train, y_train))

# ---------------------------------------------------------------------
print("\nSTEP 5 — Train Final Models")

models = {
    "logistic_regression": lr_model,
    "random_forest": rf_model,
    "gradient_boosting": gb_model,
    "stacking_ensemble": stack
}

test_metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    test_metrics[name] = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds, average="weighted"), 4)
    }
    joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")

xgb_model.fit(X_train, y_train - 1)
xgb_preds = xgb_model.predict(X_test) + 1
test_metrics["xgboost"] = {
    "accuracy": round(accuracy_score(y_test, xgb_preds), 4),
    "f1": round(f1_score(y_test, xgb_preds, average="weighted"), 4)
}
joblib.dump(xgb_model, f"{MODEL_DIR}/xgboost.pkl")

best_model = max(test_metrics, key=lambda k: test_metrics[k]["f1"])
print(f"\n★ Best model: {best_model}")

with open(f"{MODEL_DIR}/cv_results.json", "w") as f:
    json.dump({"cv": cv_results, "test": test_metrics, "tuned": TUNED}, f, indent=2)

with open(f"{MODEL_DIR}/best_model_name.txt", "w") as f:
    f.write(best_model)

print("\n✓ All models saved")
print("Next → run 03_shap_explain.py")