"""Prediction service for MP115 (hostel mess food waste).

Run locally:  PYTHONPATH=src uvicorn app.main:app --reload
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"


# ---------- JSON logging: one JSON line per event, to stdout ----------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


logger = logging.getLogger("mp115-api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
logger.propagate = False


# ---------- request schema: the client's two rules live here ----------
class MealRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")      # exactly these fields, nothing else

    students_registered: int = Field(..., ge=0)
    attendance_pct: float = Field(..., ge=0, le=100)
    menu_rating: float = Field(..., ge=0, le=5)
    temp_c: float
    cooked_kg: float = Field(..., gt=0, le=1000)   # rule 1: above 1000 kg is impossible
    plate_size_cm: float = Field(..., ge=10)       # rule 2: below 10 cm is a typing mistake
    meal: Literal["breakfast", "lunch", "dinner"]
    hostel: Literal["Boys Hostel 1", "Boys Hostel 2", "Girls Hostel"]


class Prediction(BaseModel):
    waste_kg: float
    model_version: str


app = FastAPI(title="MP115 Hostel Mess Food Waste API", version="1.0")
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict(req: MealRequest):
    start = time.perf_counter()
    row = pd.DataFrame([req.model_dump()])
    waste = max(float(model.predict(row)[0]), 0.0)   # waste cannot be negative
    logger.info("prediction", extra={"fields": {
        "event": "predict",
        "meal": req.meal,
        "hostel": req.hostel,
        "cooked_kg": req.cooked_kg,
        "prediction_kg": round(waste, 3),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }})
    return Prediction(waste_kg=round(waste, 3), model_version="champion")
