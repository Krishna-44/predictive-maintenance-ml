import numpy as np
import pytest

from predmaint.features import BASELINE_CYCLES, build_features, feature_columns, sensor_drift


def test_one_row_per_reading_and_no_missing_values(fleet):
    feats = build_features(fleet)
    assert len(feats) == len(fleet)
    assert not feats[feature_columns(feats)].isna().to_numpy().any()


@pytest.mark.parametrize("machine_id", [1, 7, 20])
@pytest.mark.parametrize("cut", [5, BASELINE_CYCLES, 60])
def test_features_are_causal(fleet, machine_id, cut):
    """Features at cycle t must not change when the future (cycles > t) is removed."""
    history = fleet[fleet["machine_id"] == machine_id]
    full = build_features(history).set_index("cycle")
    truncated = build_features(history[history["cycle"] <= cut]).set_index("cycle")
    np.testing.assert_allclose(full.loc[:cut].to_numpy(), truncated.to_numpy(), equal_nan=True)


def test_rows_are_independent_of_other_machines(fleet):
    alone = build_features(fleet[fleet["machine_id"] == 3]).reset_index(drop=True)
    together = build_features(fleet)
    together = together[together["machine_id"] == 3].reset_index(drop=True)
    np.testing.assert_allclose(alone.to_numpy(), together.to_numpy())


def test_drift_grows_towards_failure(fleet):
    machine = fleet[(fleet["machine_id"] == 2)]
    mode = machine["failure_mode"].iloc[0]
    sensor = {"bearing_wear": "vibration", "overheating": "temperature",
              "seal_leak": "pressure", "electrical_fault": "current"}[mode]
    drift = sensor_drift(machine)[sensor].abs()
    assert drift.iloc[-1] > 3 * max(drift.iloc[BASELINE_CYCLES:BASELINE_CYCLES + 10].mean(), 0.5)
