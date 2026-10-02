"""Causal feature engineering over per-machine sensor histories.

Every feature for cycle *t* uses only readings from cycles ``<= t`` of the same machine, so the
model sees exactly what it would see in production (``tests/test_features.py`` checks this by
truncating histories and comparing).

Per sensor:

* ``{s}_mean`` / ``{s}_std`` / ``{s}_slope`` over the last ``WINDOW`` cycles — level, noise, trend
* ``{s}_delta`` — rolling mean minus the machine's own healthy baseline (the mean of its first
  ``BASELINE_CYCLES`` readings). This removes machine-to-machine variation, which otherwise hides
  the degradation signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from predmaint.simulate import LOAD_FACTORS, SENSORS, SIGNALS

WINDOW = 10
BASELINE_CYCLES = 20


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "cycle"]).reset_index(drop=True)
    # Sensor dropouts: carry the last known value forward (causal), within each machine only.
    df[list(SENSORS)] = df.groupby("machine_id")[list(SENSORS)].ffill()
    return df


def _causal_baseline(values: pd.Series) -> pd.Series:
    """Mean of the first ``BASELINE_CYCLES`` readings, using only readings seen so far."""
    expanding = values.expanding().mean()
    frozen = expanding.iloc[BASELINE_CYCLES - 1] if len(values) >= BASELINE_CYCLES else np.nan
    out = expanding.copy()
    out.iloc[BASELINE_CYCLES:] = frozen
    return out


def _causal_baseline_std(values: pd.Series) -> pd.Series:
    expanding = values.expanding(min_periods=2).std()
    frozen = expanding.iloc[BASELINE_CYCLES - 1] if len(values) >= BASELINE_CYCLES else np.nan
    out = expanding.copy()
    out.iloc[BASELINE_CYCLES:] = frozen
    return out


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per input row (same order as the cleaned, sorted input)."""
    df = _clean(raw)
    grouped = df.groupby("machine_id", sort=False)
    t = df["cycle"].astype(float)

    columns: dict[str, pd.Series] = {
        "machine_id": df["machine_id"],
        "cycle": df["cycle"],
        "load_factor": df["load"].map(LOAD_FACTORS).astype(float),
    }

    def rolling(series: pd.Series, how: str) -> pd.Series:
        r = series.groupby(df["machine_id"], sort=False).rolling(WINDOW, min_periods=1)
        return getattr(r, how)().reset_index(level=0, drop=True).sort_index()

    sum_t, sum_tt = rolling(t, "sum"), rolling(t * t, "sum")
    n = rolling(t, "count")
    denom = (n * sum_tt - sum_t**2).replace(0, np.nan)

    for s in SIGNALS:
        y = df[s]
        mean = rolling(y, "mean")
        columns[f"{s}_mean"] = mean
        columns[f"{s}_std"] = rolling(y, "std").fillna(0.0)
        slope = (n * rolling(t * y, "sum") - sum_t * rolling(y, "sum")) / denom
        columns[f"{s}_slope"] = slope.fillna(0.0)
        baseline = grouped[s].transform(_causal_baseline)
        columns[f"{s}_delta"] = mean - baseline

    return pd.DataFrame(columns)


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c != "machine_id"]


def sensor_drift(raw: pd.DataFrame) -> pd.DataFrame:
    """How far each sensor's recent mean has drifted from its healthy baseline, in baseline σ.

    Used to explain an alert in plain terms ("vibration is 6.1σ above its baseline").
    """
    df = _clean(raw)
    grouped = df.groupby("machine_id", sort=False)
    out = {"machine_id": df["machine_id"], "cycle": df["cycle"]}
    for s in SENSORS:
        mean = df[s].groupby(df["machine_id"], sort=False).rolling(WINDOW, min_periods=1).mean()
        mean = mean.reset_index(level=0, drop=True).sort_index()
        base = grouped[s].transform(_causal_baseline)
        std = grouped[s].transform(_causal_baseline_std)
        out[s] = (mean - base) / std.where(std > 1e-9)
    return pd.DataFrame(out)
