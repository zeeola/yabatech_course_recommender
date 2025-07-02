"""
YABATECH Course Recommendation – Flask Service
==============================================

• loads artefacts from models/
• initialises RecommendationEngine (utils/recommendation_engine.py)
• exposes:
    GET  /health
    POST /recommend
    POST /chat
    GET  /courses
    GET  /course/<name>
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ── local imports ─────────────────────────────────────────────────────
from utils.recommendation_engine import RecommendationEngine

# ──────────────────────────────────────────────────────────────────────
# 0. Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger("app")

# ──────────────────────────────────────────────────────────────────────
# 1. Helper – course requirements loader
# ──────────────────────────────────────────────────────────────────────
def load_course_requirements(path: str = "data/course_requirements.json") -> Dict[str, Dict[str, Any]]:
    """Turn the verbose JSON into a quick-lookup dict keyed by course name."""
    with open(path, "r", encoding="utf-8") as f:
        raw: List[Dict[str, str]] = json.load(f)

    parsed: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        name = row["Program"].strip().upper()

        compulsory = [
            s.strip() for s in row["OLEVEL_Compulsory"].split(",") if s.strip().lower() != "n/a"
        ]
        elective = [
            s.strip() for s in row["OLEVEL_Elective"].split(",") if s.strip().lower() != "n/a"
        ]
        utme = [
            s.strip() for s in row["UTME_Requirement"].split(",") if s.strip().lower() != "n/a"
        ]

        parsed[name] = {
            "required_subjects": compulsory + elective,
            "utme_subjects": utme,
            "min_jamb_score": int(row["JAMB_Cutoff"]),
            "min_aggregate_score": float(row["Aggregate_Score"]),
            "min_credits": 5,
        }

    logger.info("Loaded %d course requirement entries", len(parsed))
    return parsed


# ──────────────────────────────────────────────────────────────────────
# 2. Artefact paths
# ──────────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
ARTEFACTS = {
    "model": os.path.join(MODELS_DIR, "model.pkl"),
    "scaler": os.path.join(MODELS_DIR, "scaler.pkl"),
    "label": os.path.join(MODELS_DIR, "label_encoder.pkl"),
    "waec_mlb": os.path.join(MODELS_DIR, "waec_subject_encoder.pkl"),
    "jamb_mlb": os.path.join(MODELS_DIR, "jamb_subject_encoder.pkl"),
}

# Fail fast if any artefact is missing
missing = [k for k, p in ARTEFACTS.items() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Missing artefacts {missing}. Run train_model.py before starting the server."
    )

# ──────────────────────────────────────────────────────────────────────
# 3. Engine initialisation (done once)
# ──────────────────────────────────────────────────────────────────────
engine = RecommendationEngine(
    model_path=ARTEFACTS["model"],
    scaler_path=ARTEFACTS["scaler"],
    label_path=ARTEFACTS["label"],
    waec_mlb_path=ARTEFACTS["waec_mlb"],
    jamb_mlb_path=ARTEFACTS["jamb_mlb"],
    requirements_path="data/course_requirements.json",
)

# ──────────────────────────────────────────────────────────────────────
# 4. Flask app
# ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "yct-recommender-2025"


# ——————————————————————————————————————————————————————————
# ROUTES
# ——————————————————————————————————————————————————————————
@app.route("/")
def index():
    return render_template("index.html")  # you can replace with a simple landing page


@app.route("/health")
def health():
    return jsonify(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        courses=len(engine.course_encoder.classes_),
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    payload: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    required = {"waec_subjects", "waec_grades", "jamb_subjects", "jamb_score"}
    if not required.issubset(payload):
        return (
            jsonify(
                success=False,
                error=f"Missing fields: {', '.join(sorted(required - set(payload)))}",
            ),
            400,
        )

    try:
        recs = engine.recommend(payload, top_n=5)
        return jsonify(success=True, recommendations=recs, timestamp=datetime.utcnow().isoformat())
    except Exception as e:  # pragma: no cover
        logger.exception("Recommendation failure")
        return jsonify(success=False, error=str(e)), 500


@app.route("/courses")
def courses():
    return jsonify(sorted(list(engine.course_encoder.classes_)))


@app.route("/course/<course_name>")
def course_info(course_name: str):
    info = engine.course_requirements.get(course_name.upper())
    if not info:
        return jsonify(success=False, error="Course not found"), 404
    return jsonify(success=True, course=course_name.upper(), requirements=info)


# ——————————————————————————————————————————————————————————
# CHAT BOT (very light demo)
# ——————————————————————————————————————————————————————————
@app.route("/chat", methods=["POST"])
def chat():
    txt = (request.get_json(force=True) or {}).get("message", "").lower()
    resp = "I'm not sure I understand. Ask me about course recommendations."
    if any(w in txt for w in ["hello", "hi", "hey"]):
        resp = "Hello! Send your WAEC/JAMB details to /recommend and I'll suggest courses."
    elif "course" in txt:
        resp = f"We currently support {len(engine.course_encoder.classes_)} programmes."

    return jsonify(success=True, response=resp)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("🚀  Serving on http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
