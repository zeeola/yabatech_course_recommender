# recommendation_engine.py
"""YABATECH – Hybrid Top‑5 Course Recommendation Engine
------------------------------------------------------
This module generates Top‑5 course recommendations by
combining a trained XGBoost probability model with simple
rule‑based post‑filtering.

Expected artefacts in `models/` (saved by **train_model.py**):
• model.pkl               – XGBoost (multi:softprob)
• scaler.pkl              – StandardScaler for numeric features
• label_encoder.pkl       – maps label ↔ index
• waec_subject_encoder.pkl – MultiLabelBinarizer for WAEC subjects
• jamb_subject_encoder.pkl – MultiLabelBinarizer for JAMB subjects

Features expected **in this order**:
[ one‑hot WAEC subjects ] +
[ one‑hot JAMB subjects ] +
[ WAEC_Grades_Num, JAMB_Score, JAMB_Cutoff,
  Post_UTME_Score, Aggregate_Score, Course_Cutoff ]

Author: YABATECH AI Team – 2025
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Any

import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Hybrid ML + rule filtering recommender returning Top‑N courses."""

    def __init__(
        self,
        model_path: str = "models/model.pkl",
        scaler_path: str = "models/scaler.pkl",
        label_path: str = "models/label_encoder.pkl",
        waec_mlb_path: str = "models/waec_subject_encoder.pkl",
        jamb_mlb_path: str = "models/jamb_subject_encoder.pkl",
        requirements_path: str = "data/course_requirements.json",
    ) -> None:
        # ── Load artefacts ─────────────────────────────────────────────
        self.model = joblib.load(model_path)
        self.scaler: StandardScaler = joblib.load(scaler_path)
        self.course_encoder: LabelEncoder = joblib.load(label_path)
        self.waec_mlb: MultiLabelBinarizer = joblib.load(waec_mlb_path)
        self.jamb_mlb: MultiLabelBinarizer = joblib.load(jamb_mlb_path)

        with open(requirements_path, "r", encoding="utf‑8") as f:
            self.course_requirements: Dict[str, Dict[str, Any]] = {
                d["Program"].upper(): d for d in json.load(f)
            }

        logger.info("RecommendationEngine initialised – %d courses loaded",
                    len(self.course_encoder.classes_))

    # ──────────────────────────────────────────────────────────────────
    # 1. Feature engineering helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _grade_to_num(grade: str) -> int:
        grade_map = {
            "A1": 1, "B2": 2, "B3": 3, "C4": 4, "C5": 5, "C6": 6,
            "D7": 7, "E8": 8, "F9": 9,
        }
        return grade_map.get(grade.strip().upper(), 6)

    def _numeric_waec_grade(self, waec_grades: str) -> float:
        nums = [self._grade_to_num(g) for g in waec_grades.split(",") if g.strip()]
        return float(np.mean(nums)) if nums else 6.0

    # ──────────────────────────────────────────────────────────────────
    def _vectorise_student(self, data: Dict[str, Any]) -> np.ndarray:
        """Returns a model‑ready 2‑D array (1 × n_features)."""
        # Tokenise subjects
        waec_subjects = [s.strip() for s in data["waec_subjects"].split(",") if s.strip()]
        jamb_subjects = [s.strip() for s in data["jamb_subjects"].split(",") if s.strip()]

        waec_onehot = self.waec_mlb.transform([waec_subjects])
        jamb_onehot = self.jamb_mlb.transform([jamb_subjects])

        numeric = np.array([
            self._numeric_waec_grade(data["waec_grades"]),
            data.get("jamb_score", 0),
            data.get("jamb_cutoff", 150),
            data.get("post_utme_score", 0),
            data.get("aggregate_score", 0.0),
            data.get("course_cutoff", 50.0),
        ]).reshape(1, -1)

        # Scale numeric part only
        numeric_scaled = self.scaler.transform(numeric)

        vector = np.hstack([waec_onehot, jamb_onehot, numeric_scaled])
        return vector.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────
    # 2. Core public API
    # ──────────────────────────────────────────────────────────────────
    def recommend(self, student: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """Return Top‑N course recommendations sorted by hybrid score."""
        logger.info("Generating recommendations for student…")

        # (a) Build feature vector ➜ probabilities
        X = self._vectorise_student(student)
        proba = self.model.predict_proba(X)[0]

        # (b) Pick Top‑N by probability
        top_idx = np.argsort(proba)[-top_n:][::-1]
        courses = self.course_encoder.inverse_transform(top_idx)

        recs: List[Dict[str, Any]] = []
        for course, idx in zip(courses, top_idx):
            rules = self.course_requirements.get(course.upper(), {})
            eligibility, reasons = self._rule_filter(student, rules)
            recs.append({
                "course": course,
                "ml_probability": float(proba[idx]),
                "eligible": eligibility,
                "eligibility_reasons": reasons,
            })

        # (c) Optional: sort by ML prob but keep eligible first
        recs.sort(key=lambda d: (not d["eligible"], -d["ml_probability"]))
        return recs

    # ──────────────────────────────────────────────────────────────────
    # 3. Rule‑based eligibility filter
    # ──────────────────────────────────────────────────────────────────
    def _rule_filter(self, student: Dict[str, Any], rules: Dict[str, Any]):
        if not rules:
            return True, []

        reasons = []
        jamb_score = student.get("jamb_score", 0)
        if jamb_score < int(rules.get("JAMB_Cutoff", 150)):
            reasons.append("JAMB below cutoff")

        # Simple core‑subject check
        compulsory = [s.strip().lower() for s in rules.get("OLEVEL_Compulsory", "").split(",")]
        waec_subjects = [s.strip().lower() for s in student["waec_subjects"].split(",")]
        if not all(any(c in w for w in waec_subjects) for c in compulsory):
            reasons.append("Missing compulsory WAEC subjects")

        return len(reasons) == 0, reasons


# ─────────────────────────────────────────────────────────────────────
# Convenience factory (optional)
# ─────────────────────────────────────────────────────────────────────

def load_engine() -> RecommendationEngine:
    return RecommendationEngine()
