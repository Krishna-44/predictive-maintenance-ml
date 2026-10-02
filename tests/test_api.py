import pytest
from fastapi.testclient import TestClient

from predmaint.api import create_app
from predmaint.service import demo_fleet

SENSOR_FIELDS = [
    "duty", "temperature", "vibration", "pressure", "rpm", "torque", "current", "humidity", "voltage",
]


@pytest.fixture(scope="module")
def client(bundle):
    return TestClient(create_app(bundle=bundle, fleet=demo_fleet(n_machines=6)))


def readings(fleet, machine_id, upto=None):
    rows = fleet[fleet["machine_id"] == machine_id]
    if upto is not None:
        rows = rows[rows["cycle"] <= upto]
    out = []
    for _, r in rows.iterrows():
        item = {"cycle": int(r["cycle"]), "load": r["load"]}
        item.update({f: (None if r[f] != r[f] else float(r[f])) for f in SENSOR_FIELDS})
        out.append(item)
    return out


def test_health_and_metrics(client, bundle):
    assert client.get("/api/health").json()["threshold"] == bundle.threshold
    assert "classifier" in client.get("/api/metrics").json()


def test_fleet_endpoint(client):
    fleet = client.get("/api/fleet").json()
    assert len(fleet) == 6
    expected = {"machine_id", "status", "failure_probability", "rul_estimate", "drivers", "risk_trend"}
    assert expected <= set(fleet[0])


def test_predict_matches_the_service(client, fleet):
    res = client.post("/api/predict", json={"machine_id": 9, "readings": readings(fleet, 9, upto=80)})
    assert res.status_code == 200
    body = res.json()
    assert body["machine_id"] == 9 and body["cycle"] == 80
    assert 0 <= body["failure_probability"] <= 1


def test_predict_handles_missing_sensor_values(client, fleet):
    rows = readings(fleet, 4, upto=40)
    for r in rows:
        r["humidity"] = None
    assert client.post("/api/predict", json={"readings": rows}).status_code == 200


def test_predict_validates_input(client, fleet):
    assert client.post("/api/predict", json={"readings": []}).status_code == 422
    bad_load = [{"cycle": 1, "load": "extreme"}]
    assert client.post("/api/predict", json={"readings": bad_load}).status_code == 422
    dupes = readings(fleet, 4, upto=3)
    dupes[1]["cycle"] = dupes[0]["cycle"]
    assert client.post("/api/predict", json={"readings": dupes}).status_code == 422


def test_dashboard_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "Fleet Health" in res.text
