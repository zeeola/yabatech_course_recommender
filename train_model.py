# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import os

# Load dataset
df = pd.read_csv("data/synthetic_dataset_100k.csv")

# Encode categorical features
def preprocess(df):
    df = df.copy()
    # WAEC subjects
    mlb_subjects = MultiLabelBinarizer()
    waec_subjects = mlb_subjects.fit_transform(df["WAEC_Subjects"].str.split(", "))
    
    # WAEC grades as ordinal numbers
    grade_map = {"A1": 1, "B2": 2, "B3": 3, "C4": 4, "C5": 5, "C6": 6, "D7": 7, "E8": 8, "F9": 9}
    df["WAEC_Grades_Num"] = df["WAEC_Grades"].apply(lambda x: np.mean([grade_map[g.strip()] for g in x.split(", ")]))

    # JAMB Subjects
    mlb_jamb = MultiLabelBinarizer()
    jamb_subjects = mlb_jamb.fit_transform(df["JAMB_Subjects"].str.split(", "))

    # Combine features
    X = np.hstack([
        waec_subjects,
        jamb_subjects,
        df[["WAEC_Grades_Num", "JAMB_Score", "JAMB_Cutoff", "Post_UTME_Score", "Aggregate_Score", "Course_Cutoff"]].values
    ])

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df["Admitted_Course"])

    return X, y, le, mlb_subjects, mlb_jamb

X, y, le, mlb_subj, mlb_jamb = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softmax',
    max_depth=10,
    n_estimators=100,
    learning_rate=0.1,
    tree_method='hist',
    verbosity=1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model (keep <20MB)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(mlb_subj, "models/waec_subject_encoder.pkl")
joblib.dump(mlb_jamb, "models/jamb_subject_encoder.pkl")
print("✅ Model saved to models/ folder")
