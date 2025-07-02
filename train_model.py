# train_model.py
"""
Trains an XGBoost multi-class model and saves artefacts under ./models
Artefacts:  model.pkl, scaler.pkl, label_encoder.pkl,
            waec_subject_encoder.pkl, jamb_subject_encoder.pkl
The entire folder is <20 MB with default params.
"""
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler

DATA_CSV = "data/synthetic_dataset_100k.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────
# 1. Load & basic preprocessing
# ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV)

grade_map = {"A1": 1, "B2": 2, "B3": 3, "C4": 4, "C5": 5, "C6": 6, "D7": 7, "E8": 8, "F9": 9}
df["WAEC_Grades_Num"] = df["WAEC_Grades"].apply(
    lambda s: np.mean([grade_map.get(g.strip(), 6) for g in s.split(",")])
)

waec_mlb = MultiLabelBinarizer()
waec_onehot = waec_mlb.fit_transform(df["WAEC_Subjects"].str.split(", "))

jamb_mlb = MultiLabelBinarizer()
jamb_onehot = jamb_mlb.fit_transform(df["JAMB_Subjects"].str.split(", "))

numeric = df[
    ["WAEC_Grades_Num", "JAMB_Score", "JAMB_Cutoff", "Post_UTME_Score", "Aggregate_Score", "Course_Cutoff"]
].values
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric)

X = np.hstack([waec_onehot, jamb_onehot, numeric_scaled])

le = LabelEncoder()
y = le.fit_transform(df["Admitted_Course"])

# ───────────────────────────────────────────────────────────────────
# 2. Train / test
# ───────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    max_depth=8,
    n_estimators=140,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=le.classes_))

# ───────────────────────────────────────────────────────────────────
# 3. Save artefacts
# ───────────────────────────────────────────────────────────────────
joblib.dump(model, os.path.join(OUT_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.pkl"))
joblib.dump(waec_mlb, os.path.join(OUT_DIR, "waec_subject_encoder.pkl"))
joblib.dump(jamb_mlb, os.path.join(OUT_DIR, "jamb_subject_encoder.pkl"))
print("✅  Artefacts saved to", OUT_DIR)
