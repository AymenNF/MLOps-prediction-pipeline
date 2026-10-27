"""Prediction API.

Run with:  uvicorn mlops_pipeline.serve:app --reload
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mlops_pipeline.config import ROOT, load_params
from mlops_pipeline.data import FEATURES

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / load_params()["paths"]["model"]))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="MLOps pipeline - breast cancer classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    # One dict per sample, keyed by feature name (see GET /features).
    instances: list[dict[str, float]]


class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/features")
def features() -> list[str]:
    return FEATURES


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = state.get("model")
    if model is None:
        raise HTTPException(503, "Model not loaded. Run `python -m mlops_pipeline.train` first.")
    if not req.instances:
        raise HTTPException(422, "instances must not be empty")
    missing = set(FEATURES) - set(req.instances[0])
    if missing:
        raise HTTPException(422, f"Missing features: {sorted(missing)}")

    X = pd.DataFrame(req.instances)[FEATURES]
    return PredictResponse(
        predictions=model.predict(X).tolist(),
        probabilities=model.predict_proba(X)[:, 1].round(4).tolist(),
    )
