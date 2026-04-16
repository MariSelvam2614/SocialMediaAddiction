"""
FILE: 01_data_fusion.py
IMPROVEMENT: Better Data & Features
- Combines SMA.csv + Students_Social_Media_Addiction.csv
- Engineers new features (behavioral intensity, interaction terms)
- Handles class imbalance with SMOTE
- Saves unified dataset ready for training
Run: python 01_data_fusion.py
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

DATA_DIR  = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── LOAD BOTH DATASETS ───────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 1 — Loading & Fusing Both Datasets")
print("=" * 60)

sma = pd.read_csv(f"{DATA_DIR}/SMA.csv")
stu = pd.read_csv(f"{DATA_DIR}/Students_Social_Media_Addiction.csv")

print(f"  SMA.csv          : {sma.shape[0]} rows")
print(f"  Students.csv     : {stu.shape[0]} rows")

# ─── HARMONISE SMA.CSV ────────────────────────────────────────────────────────
sma.rename(columns={
    "Q1_Gender": "Gender", "Q2_Living_Area": "Living_Area",
    "Q3_Maritial_Status": "Marital_Status",
    "SMAQ1": "Time_Spent",       "SMAQ2": "Usage_Frequency",
    "SMAQ3": "Purpose_Use",      "SMAQ4": "Neglect_Duties",
    "SMAQ5": "Preoccupation",    "SMAQ6": "Withdrawal_Anxiety",
    "SMAQ7": "Attempted_Control","SMAQ8": "Mood_Regulation",
    "SMAQ9": "Deception_Use",    "SMAQ10": "Relationship_Impact",
    "SMA_Scale_value": "Total_Score",
    "SMA_Scale (Class_Lebel)": "Risk_Class"
}, inplace=True)

# Map SMA gender: 1=Male, 2=Female → 0/1
sma["Gender_enc"] = sma["Gender"] - 1

# ─── HARMONISE STUDENTS.CSV ───────────────────────────────────────────────────
# Encode categoricals
stu["Gender_enc"]    = (stu["Gender"] == "Female").astype(int)
stu["Academic_enc"]  = stu["Academic_Level"].map({"High School": 1, "Undergraduate": 2, "Graduate": 3})
stu["Relation_enc"]  = stu["Relationship_Status"].map({"Single": 1, "In Relationship": 2, "Complicated": 3})
stu["AcadImpact_enc"]= (stu["Affects_Academic_Performance"] == "Yes").astype(int)
platform_map = {p: i+1 for i, p in enumerate(stu["Most_Used_Platform"].unique())}
stu["Platform_enc"]  = stu["Most_Used_Platform"].map(platform_map)

# Derive Likert-like features from continuous variables (1–5 scale)
stu["Time_Spent"]          = pd.cut(stu["Avg_Daily_Usage_Hours"],  bins=[0,1,2,4,6,24], labels=[1,2,3,4,5]).astype(float)
stu["Withdrawal_Anxiety"]  = (6 - pd.cut(stu["Mental_Health_Score"],   bins=[0,2,4,6,8,10], labels=[1,2,3,4,5]).astype(float))
stu["Relationship_Impact"] = pd.cut(stu["Conflicts_Over_Social_Media"], bins=[-1,0,1,2,3,20], labels=[1,2,3,4,5]).astype(float)
stu["Neglect_Duties"]      = stu["AcadImpact_enc"] * 3 + 1   # proxy: 1 if no impact, 4 if yes
stu["Mood_Regulation"]     = (6 - pd.cut(stu["Sleep_Hours_Per_Night"], bins=[0,5,6,7,8,12], labels=[1,2,3,4,5]).astype(float))

# Derive target class from Addicted_Score (2–9)
def score_to_class(s):
    if s <= 4: return 1
    if s <= 6: return 2
    return 3

stu["Risk_Class"] = stu["Addicted_Score"].apply(score_to_class)

# Fill remaining Likert columns with median (3 = neutral)
for col in ["Usage_Frequency","Purpose_Use","Preoccupation","Attempted_Control","Deception_Use"]:
    stu[col] = 3.0   # neutral default for missing Likert items

stu["Living_Area"]   = 1   # default urban (not in students dataset)
stu["Marital_Status"]= stu["Relation_enc"]

# ─── UNIFY FEATURE SET ────────────────────────────────────────────────────────
CORE_FEATURES = [
    "Gender_enc", "Living_Area", "Marital_Status",
    "Time_Spent", "Usage_Frequency", "Purpose_Use",
    "Neglect_Duties", "Preoccupation", "Withdrawal_Anxiety",
    "Attempted_Control", "Mood_Regulation", "Deception_Use",
    "Relationship_Impact"
]

sma_unified = sma[CORE_FEATURES + ["Risk_Class"]].copy()
stu_unified = stu[CORE_FEATURES + ["Risk_Class"]].copy()

# Drop rows with NaN (from cuts)
sma_unified.dropna(inplace=True)
stu_unified.dropna(inplace=True)

print(f"\n  SMA after processing : {sma_unified.shape[0]} rows")
print(f"  Students after proc  : {stu_unified.shape[0]} rows")

# ─── MERGE ────────────────────────────────────────────────────────────────────
unified = pd.concat([sma_unified, stu_unified], ignore_index=True)
unified = unified.astype(float)
unified["Risk_Class"] = unified["Risk_Class"].astype(int)

print(f"\n  UNIFIED dataset      : {unified.shape[0]} rows × {unified.shape[1]} cols")
print(f"  Class distribution:")
print(unified["Risk_Class"].value_counts().sort_index().to_string())

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
print("\n  STEP 2 — Feature Engineering")

X = unified[CORE_FEATURES].copy()
y = unified["Risk_Class"].copy()

# 1. Behavioral Intensity Score (mean of 10 Likert items)
likert_items = ["Time_Spent","Usage_Frequency","Purpose_Use","Neglect_Duties",
                "Preoccupation","Withdrawal_Anxiety","Attempted_Control",
                "Mood_Regulation","Deception_Use","Relationship_Impact"]
X["Behavioral_Intensity"] = X[likert_items].mean(axis=1)

# 2. Withdrawal × Neglect (high on both = strong addiction signal)
X["Withdrawal_x_Neglect"] = X["Withdrawal_Anxiety"] * X["Neglect_Duties"]

# 3. Control Failure Index (low Attempted_Control + high Time_Spent)
X["Control_Failure"]      = X["Time_Spent"] * (6 - X["Attempted_Control"])

# 4. Social Damage Score (relationship + deception)
X["Social_Damage"]        = X["Relationship_Impact"] + X["Deception_Use"]

# 5. Craving Score (preoccupation + withdrawal)
X["Craving_Score"]        = X["Preoccupation"] + X["Withdrawal_Anxiety"]

print(f"  Original features  : {len(CORE_FEATURES)}")
print(f"  Engineered features: 5 new")
print(f"  Total features     : {X.shape[1]}")

ENGINEERED_FEATURES = CORE_FEATURES + [
    "Behavioral_Intensity","Withdrawal_x_Neglect",
    "Control_Failure","Social_Damage","Craving_Score"
]

# ─── SMOTE — Fix Class Imbalance ──────────────────────────────────────────────
print("\n  STEP 3 — SMOTE Oversampling (fix class imbalance)")
print(f"  Before SMOTE: {y.value_counts().sort_index().to_dict()}")

sm = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = sm.fit_resample(X, y)

print(f"  After  SMOTE: {pd.Series(y_resampled).value_counts().sort_index().to_dict()}")
print(f"  Total samples after SMOTE: {len(y_resampled)}")

# ─── SCALE ────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
X_scaled_df = pd.DataFrame(X_scaled, columns=ENGINEERED_FEATURES)

# ─── SAVE ─────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_resampled,
    test_size=0.2, random_state=42, stratify=y_resampled
)

X_train.to_csv(f"{DATA_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{DATA_DIR}/X_test.csv",   index=False)
pd.Series(y_train, name="Risk_Class").to_csv(f"{DATA_DIR}/y_train.csv", index=False)
pd.Series(y_test,  name="Risk_Class").to_csv(f"{DATA_DIR}/y_test.csv",  index=False)
unified.to_csv(f"{DATA_DIR}/unified_dataset.csv", index=False)

joblib.dump(scaler,              f"{MODEL_DIR}/scaler.pkl")
joblib.dump(ENGINEERED_FEATURES, f"{MODEL_DIR}/feature_cols.pkl")
joblib.dump(platform_map,        f"{MODEL_DIR}/platform_map.pkl")

print(f"\n  ✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"  ✓ Saved: X_train, X_test, y_train, y_test, unified_dataset")
print(f"  ✓ Saved: scaler.pkl, feature_cols.pkl")
print("\n  Next → run 02_train_improved.py")
