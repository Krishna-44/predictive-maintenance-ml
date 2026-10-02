import pytest

from predmaint.service import MaintenanceAdvisor, demo_fleet, load_bundle, load_or_train, save_bundle


def test_assessment_near_failure_raises_an_alert(bundle, fleet):
    advisor = MaintenanceAdvisor(bundle)
    history = fleet[fleet["machine_id"] == 5]
    early = advisor.assess_history(history[history["cycle"] <= 30])
    late = advisor.assess_history(history[history["rul"] >= 3])
    assert early["status"] == "ok"
    assert late["failure_probability"] > early["failure_probability"]
    assert late["status"] == "alert"
    assert late["rul_estimate"] < early["rul_estimate"]
    assert len(late["drivers"]) == 3 and abs(late["drivers"][0]["z"]) >= abs(late["drivers"][-1]["z"])


def test_status_rules(bundle):
    advisor = MaintenanceAdvisor(bundle)
    t, h = bundle.threshold, bundle.horizon
    assert advisor.status(t, 0, 100) == "alert"
    assert advisor.status(t / 2, 0, 100) == "watch"
    assert advisor.status(0, 4, 100) == "watch"
    assert advisor.status(0, 0, 2 * h) == "watch"
    assert advisor.status(0, 0, 100) == "ok"


def test_assessment_requires_a_single_machine(bundle, fleet):
    with pytest.raises(ValueError):
        MaintenanceAdvisor(bundle).assess_history(fleet[fleet["machine_id"] <= 2])


def test_fleet_is_sorted_by_urgency(bundle):
    results = MaintenanceAdvisor(bundle).assess_fleet(demo_fleet(n_machines=8))
    order = {"alert": 0, "watch": 1, "ok": 2}
    keys = [(order[r["status"]], -r["failure_probability"]) for r in results]
    assert keys == sorted(keys)
    assert set(results[0]["trends"]) == {"vibration", "temperature", "pressure", "current"}


def test_demo_fleet_hides_labels():
    assert {"rul", "failure_mode"}.isdisjoint(demo_fleet(n_machines=3).columns)


def test_bundle_round_trips_through_disk(bundle, tmp_path):
    path = tmp_path / "model.joblib"
    save_bundle(bundle, path)
    assert load_bundle(path).threshold == bundle.threshold
    assert load_or_train(path).features == bundle.features
