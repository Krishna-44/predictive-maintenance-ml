"""FastAPI service: single-machine predictions and a fleet dashboard.

Run with ``predmaint serve`` or ``uvicorn predmaint.api:create_app --factory``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from predmaint.service import MaintenanceAdvisor, demo_fleet, load_or_train
from predmaint.train import ModelBundle

STATIC_DIR = Path(__file__).parent / "static"


class Reading(BaseModel):
    cycle: int = Field(ge=1)
    load: Literal["low", "medium", "high"]
    duty: float | None = None
    temperature: float | None = None
    vibration: float | None = None
    pressure: float | None = None
    rpm: float | None = None
    torque: float | None = None
    current: float | None = None
    humidity: float | None = None
    voltage: float | None = None


class PredictRequest(BaseModel):
    machine_id: int = 0
    readings: list[Reading] = Field(min_length=1, max_length=5000)


def create_app(bundle: ModelBundle | None = None, fleet: pd.DataFrame | None = None) -> FastAPI:
    bundle = bundle or load_or_train(os.environ.get("PREDMAINT_MODEL", "artifacts/model.joblib"))
    advisor = MaintenanceAdvisor(bundle)
    fleet = demo_fleet() if fleet is None else fleet

    app = FastAPI(
        title="Predictive Maintenance API",
        version="0.1.0",
        description="Failure risk, remaining useful life and anomaly scores from machine telemetry.",
    )

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "threshold": bundle.threshold, "horizon_cycles": bundle.horizon}

    @app.get("/api/metrics")
    def metrics() -> dict:
        return bundle.metrics

    @app.get("/api/fleet")
    def fleet_status() -> list[dict]:
        return advisor.assess_fleet(fleet)

    @app.post("/api/predict")
    def predict(req: PredictRequest) -> dict:
        history = pd.DataFrame([r.model_dump() for r in req.readings]).assign(machine_id=req.machine_id)
        if history["cycle"].duplicated().any():
            raise HTTPException(422, "cycles must be unique")
        history = history.sort_values("cycle").astype({c: float for c in history.columns
                                                       if c not in ("cycle", "load", "machine_id")})
        return advisor.assess_history(history)

    return app
