"""
FILE: 03_shap_explain.py
IMPROVEMENT: SHAP Explainability
- Global feature importance via SHAP
- Per-prediction local explanation
- Saves SHAP plots + explainer object
Run: python 03_shap_explain.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib, os, warnings
warnings.filterwarnings("ignore")

DATA_DIR   = "data"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs/shap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0b0f1a", "axes.facecolor": "#111827",
    "text.color": "#e2e8f0", "axes.labelcolor": "#e2e8f0",
    "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
    "grid.color": "#1f2d45",
})

# ─── LOAD ─────────────────────────────────────────────────────────────────────
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")

# Use Random Forest for SHAP (tree-based, fastest)
rf = joblib.load(f"{MODEL_DIR}/random_forest.pkl")

try:
    import shap

    print("=" * 60)
    print("  SHAP Explainability Analysis")
    print("=" * 60)

    # ── GLOBAL EXPLAINER ──────────────────────────────────────────
    print("  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)   # shape: (n_samples, n_features, n_classes)

    # Save explainer for use in API
    joblib.dump(explainer, f"{MODEL_DIR}/shap_explainer.pkl")
    print("  ✓ Saved shap_explainer.pkl")

    # ── PLOT 1 : Summary Bar — Class averaged ─────────────────────
    # Average absolute SHAP across all classes
    shap_abs_mean = np.mean([np.abs(shap_values[:,:,i]) for i in range(3)], axis=0)
    importance_df = pd.DataFrame(shap_abs_mean, columns=feature_cols)
    global_imp = importance_df.mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(global_imp)))
    bars = ax.barh(global_imp.index, global_imp.values, color=colors, edgecolor="#0b0f1a")
    ax.set_title("SHAP Feature Importance (Global)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP Value|")
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.3f}", va="center", fontsize=8, color="#e2e8f0")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_shap_global.png", dpi=150)
    plt.close()
    print("  ✓ Saved 01_shap_global.png")

    # ── PLOT 2 : Per-class SHAP bars ─────────────────────────────
    class_names = ["Low", "Moderate", "High"]
    colors_cls  = ["#34d399", "#fbbf24", "#f87171"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for i, (cls_name, color) in enumerate(zip(class_names, colors_cls)):
        imp = pd.Series(np.abs(shap_values[:,:,i]).mean(axis=0),
                        index=feature_cols).sort_values(ascending=True)
        axes[i].barh(imp.index, imp.values, color=color, alpha=0.85, edgecolor="#0b0f1a")
        axes[i].set_title(f"SHAP — {cls_name} Class", fontsize=11, fontweight="bold", color="#e2e8f0")
        axes[i].set_xlabel("Mean |SHAP|")
    plt.suptitle("Per-Class SHAP Feature Importance", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_shap_per_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved 02_shap_per_class.png")

    # ── LOCAL EXPLANATION FUNCTION ────────────────────────────────
    def explain_prediction(input_row_df: pd.DataFrame) -> dict:
        """
        Given a single-row DataFrame of features (already scaled),
        return a dict of {feature: shap_value} for the predicted class.
        """
        sv = explainer.shap_values(input_row_df)      # list of 3 arrays
        pred_class = rf.predict(input_row_df)[0]       # 1, 2, or 3
        class_idx  = pred_class - 1

        feature_shap = dict(zip(feature_cols, sv[class_idx][0]))
        sorted_shap  = dict(sorted(feature_shap.items(),
                                    key=lambda x: abs(x[1]), reverse=True))
        top_positive = [(k,v) for k,v in sorted_shap.items() if v > 0][:5]
        top_negative = [(k,v) for k,v in sorted_shap.items() if v < 0][:3]

        return {
            "predicted_class": int(pred_class),
            "all_shap":        {k: round(float(v), 4) for k,v in sorted_shap.items()},
            "top_drivers":     [{"feature": k, "shap": round(float(v),4),
                                  "direction": "increases_risk"} for k,v in top_positive],
            "top_reducers":    [{"feature": k, "shap": round(float(v),4),
                                  "direction": "reduces_risk"}  for k,v in top_negative],
        }

    # Save the function reference via a wrapper class
    class SHAPExplainerWrapper:
        def __init__(self, explainer, feature_cols, model):
            self.explainer    = explainer
            self.feature_cols = feature_cols
            self.model        = model

        def explain(self, input_row_df):
            return explain_prediction(input_row_df)

    wrapper = SHAPExplainerWrapper(explainer, feature_cols, rf)
    joblib.dump(wrapper, f"{MODEL_DIR}/shap_wrapper.pkl")

    # ── SAMPLE LOCAL EXPLANATION ─────────────────────────────────
    print("\n  Sample local explanation (first test row):")
    sample = X_test.iloc[[0]]
    local_exp = explain_prediction(sample)
    print(f"    Predicted class: {local_exp['predicted_class']}")
    print(f"    Top risk drivers:")
    for d in local_exp["top_drivers"][:3]:
        print(f"      {d['feature']:<30} SHAP={d['shap']:+.4f}")

    print("\n  ✓ SHAP analysis complete")
    SHAP_OK = True

except ImportError:
    print("  ⚠ SHAP not installed. Run: pip install shap")
    print("    Falling back to Random Forest feature_importances_")
    SHAP_OK = False

    # Fallback: standard feature importance
    rf = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(10,7))
    ax.barh(imp.index, imp.values, color="#60a5fa", edgecolor="#0b0f1a")
    ax.set_title("Feature Importance (RF Gini)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_feature_importance_fallback.png", dpi=150)
    plt.close()
    print("  ✓ Fallback importance plot saved")

print(f"\n  SHAP plots → {OUTPUT_DIR}/")
print("  Next → run 04_app_improved.py")
