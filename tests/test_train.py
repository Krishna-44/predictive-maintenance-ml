import numpy as np

from predmaint.train import CostModel, choose_threshold


def test_cost_model_counts_misses_and_false_alarms():
    y = np.array([1, 1, 0, 0])
    alerts = np.array([True, False, True, False])
    assert CostModel(missed_failure=10, false_alarm=1).expected_cost(y, alerts) == (10 + 1) / 4


def test_threshold_moves_with_the_cost_of_a_miss():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.1
    proba = np.clip(np.where(y, 0.6, 0.2) + rng.normal(0, 0.15, 2000), 0, 1)
    cheap_miss = choose_threshold(y.astype(int), proba, CostModel(missed_failure=1))
    costly_miss = choose_threshold(y.astype(int), proba, CostModel(missed_failure=50))
    assert costly_miss < cheap_miss


def test_trained_bundle_beats_chance_on_unseen_machines(bundle):
    m = bundle.metrics
    assert m["data"]["train_machines"] + m["data"]["test_machines"] == m["data"]["machines"]
    assert m["classifier"]["roc_auc"] > 0.9
    assert m["classifier"]["pr_auc"] > 3 * m["data"]["positive_rate"]
    assert m["rul"]["mae"] < 30
    assert 0 < bundle.threshold < 1


def test_bundle_predictions_are_well_formed(bundle, fleet):
    from predmaint.features import build_features

    preds = bundle.predict(build_features(fleet[fleet["machine_id"] <= 3]))
    assert list(preds.columns) == ["failure_probability", "alert", "rul_estimate", "anomaly_z"]
    assert preds["failure_probability"].between(0, 1).all()
    assert (preds["rul_estimate"] >= 0).all()
