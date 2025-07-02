# utils/recommendation_engine.py
"""Hybrid Top-N Course Recommender for YABATECH."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """ML probability + rule eligibility recommender."""

    _GRADE_MAP = {
        "A1": 1,
        "B2": 2,
        "B3": 3,
        "C4": 4,
        "C5": 5,
        "C6": 6,
        "D7": 7,
        "E8": 8,
        "F9": 9,
    }

    # ───────────────────────────────────────────────────────────────
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        label_path: str,
        waec_mlb_path: str,
        jamb_mlb_path: str,
        requirements_path: str,
    ) -> None:
        self.model = joblib.load(model_path)
        self.scaler: StandardScaler = joblib.load(scaler_path)
        self.label_encoder: LabelEncoder = joblib.load(label_path)
        self.waec_mlb: MultiLabelBinarizer = joblib.load(waec_mlb_path)
        self.jamb_mlb: MultiLabelBinarizer = joblib.load(jamb_mlb_path)

        with open(requirements_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.course_requirements: Dict[str, Dict[str, Any]] = {r["Program"].upper(): r for r in raw}

        logger.info("Engine loaded: %d courses", len(self.label_encoder.classes_))

    # ───────────────────────────────────────────────────────────────
    # 1. Feature engineering
    # ───────────────────────────────────────────────────────────────
    def _waec_numeric(self, grades_str: str) -> float:
        nums = [
            self._GRADE_MAP.get(g.strip().upper(), 6)
            for g in grades_str.split(",")
            if g.strip()
        ]
        return float(np.mean(nums)) if nums else 6.0

    def _vectorise(self, student: Dict[str, Any]) -> np.ndarray:
        """Return 1×n feature vector aligned with training pipeline."""
        waec_subj = [s.strip() for s in student["waec_subjects"].split(",") if s.strip()]
        jamb_subj = [s.strip() for s in student["jamb_subjects"].split(",") if s.strip()]

        waec_onehot = self.waec_mlb.transform([waec_subj])
        jamb_onehot = self.jamb_mlb.transform([jamb_subj])

        num = np.array(
            [
                self._waec_numeric(student["waec_grades"]),
                student.get("jamb_score", 0),
                student.get("jamb_cutoff", 150),
                student.get("post_utme_score", 0),
                student.get("aggregate_score", 0),
                student.get("course_cutoff", 50),
            ]
        ).reshape(1, -1)
        num_scaled = self.scaler.transform(num)

        return np.hstack([waec_onehot, jamb_onehot, num_scaled]).astype(np.float32)

    # ───────────────────────────────────────────────────────────────
    # 2. Eligibility (very lightweight rules)
    # ───────────────────────────────────────────────────────────────
    def _eligible(self, course: str, student: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Return eligibility flag and list of reasons if not eligible."""
        rules = self.course_requirements.get(course.upper())
        if not rules:
            return True, []

        reasons: List[str] = []
        if student.get("jamb_score", 0) < int(rules["JAMB_Cutoff"]):
            reasons.append("JAMB below cutoff")

        waec_set = {s.strip().lower() for s in student["waec_subjects"].split(",")}
        for comp in [
            s.strip().lower() for s in rules["OLEVEL_Compulsory"].split(",") if s.strip()
        ]:
            if comp not in waec_set:
                reasons.append("Missing compulsory subject")
                break

        return len(reasons) == 0, reasons

    # ───────────────────────────────────────────────────────────────
    # 3. Public API
    # ───────────────────────────────────────────────────────────────
    def recommend(self, student: Dict[str, Any], *, top_n: int = 5) -> List[Dict[str, Any]]:
        """Return a list of Top-N course dicts sorted by ML probability / eligibility."""
        X = self._vectorise(student)
        proba = self.model.predict_proba(X)[0]  # type: ignore[attr-defined]

        idx = np.argsort(proba)[-top_n:][::-1]
        courses = self.label_encoder.inverse_transform(idx)

        recs: List[Dict[str, Any]] = []
        for c, i in zip(courses, idx):
            eligible, reasons = self._eligible(c, student)
            recs.append(
                {
                    "course": c,
                    "ml_probability": round(float(proba[i]), 4),
                    "eligible": eligible,
                    "reasons": reasons,
                }
            )

        # Prioritise eligible courses first, then by probability
        recs.sort(key=lambda d: (not d["eligible"], -d["ml_probability"]))
        return recs
