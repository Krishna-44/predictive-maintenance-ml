"""Turning model outputs into maintenance decisions for one machine or a whole fleet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from predmaint.features import build_features, sensor_drift
from predmaint.simulate import INFORMATIVE_SENSORS, SimulationConfig, simulate_fleet
from predmaint.train import ModelBundle, train

DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")
TREND_CYCLES = 60


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> ModelBundle:
    # joblib uses pickle: only load model files you produced yourself.
    return joblib.load(path)


def load_or_train(path: str | Path = DEFAULT_MODEL_PATH) -> ModelBundle:
    path = Path(path)
    if path.exists():
        return load_bundle(path)
    bundle = train(simulate_fleet(), with_diagnostics=False)
    save_bundle(bundle, path)
    return bundle


class MaintenanceAdvisor:
    """Wraps a trained bundle: per-machine risk, status and a plain-language explanation."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def status(self, probability: float, anomaly_z: float, rul_estimate: float) -> str:
        """``alert`` → schedule maintenance; ``watch`` → any model sees early signs of trouble."""
        if probability >= self.bundle.threshold:
            return "alert"
        near_end = rul_estimate <= 2 * self.bundle.horizon
        if probability >= self.bundle.threshold / 3 or anomaly_z >= 3 or near_end:
            return "watch"
        return "ok"

    def assess_history(self, history: pd.DataFrame) -> dict[str, Any]:
        """Assess one machine from its readings since commissioning (latest cycle last)."""
        if history["machine_id"].nunique() != 1:
            raise ValueError("history must contain exactly one machine")
        preds = self.bundle.predict(build_features(history))
        drift = sensor_drift(history).iloc[-1]
        last = preds.iloc[-1]
        drivers = sorted(
            (
                {"sensor": s, "z": round(float(drift[s]), 2)}
                for s in INFORMATIVE_SENSORS
                if np.isfinite(drift[s])
            ),
            key=lambda d: -abs(d["z"]),
        )[:3]
        prob = float(last["failure_probability"])
        anomaly_z = float(last["anomaly_z"])
        rul = float(last["rul_estimate"])
        return {
            "machine_id": int(history["machine_id"].iloc[0]),
            "cycle": int(history["cycle"].max()),
            "failure_probability": round(prob, 4),
            "status": self.status(prob, anomaly_z, rul),
            "rul_estimate": round(rul, 1),
            "anomaly_z": round(anomaly_z, 2),
            "drivers": drivers,
            "risk_trend": [round(float(p), 4) for p in preds["failure_probability"].tail(TREND_CYCLES)],
        }

    def assess_fleet(self, fleet: pd.DataFrame) -> list[dict[str, Any]]:
        results = []
        for _, history in fleet.groupby("machine_id"):
            result = self.assess_history(history)
            tail = history.sort_values("cycle").tail(TREND_CYCLES)
            result["load"] = str(history["load"].iloc[0])
            result["trends"] = {
                s: [None if pd.isna(v) else round(float(v), 3) for v in tail[s]]
                for s in ("vibration", "temperature", "pressure", "current")
            }
            results.append(result)
        order = {"alert": 0, "watch": 1, "ok": 2}
        return sorted(results, key=lambda r: (order[r["status"]], -r["failure_probability"]))


def demo_fleet(n_machines: int = 24, seed: int = 2026) -> pd.DataFrame:
    """A fresh fleet (unseen during training), each machine observed part-way through its life."""
    fleet = simulate_fleet(SimulationConfig(n_machines=n_machines, seed=seed))
    rng = np.random.default_rng(seed)
    parts = []
    for _, history in fleet.groupby("machine_id"):
        life = int(history["cycle"].max())
        # Skew towards late life so the dashboard shows a mix of healthy and degrading machines.
        cut = int(np.clip(life * rng.beta(4, 1.3), 30, life))
        parts.append(history[history["cycle"] <= cut])
    return pd.concat(parts, ignore_index=True).drop(columns=["rul", "failure_mode"])
