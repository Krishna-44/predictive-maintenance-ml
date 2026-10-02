import numpy as np

from predmaint.simulate import FAILURE_MODES, SENSORS, SimulationConfig, simulate_fleet


def test_simulation_is_deterministic_per_seed():
    a = simulate_fleet(SimulationConfig(n_machines=5, seed=3))
    b = simulate_fleet(SimulationConfig(n_machines=5, seed=3))
    c = simulate_fleet(SimulationConfig(n_machines=5, seed=4))
    assert a.equals(b)
    assert not a.equals(c)


def test_every_machine_runs_to_failure(fleet):
    for _, g in fleet.groupby("machine_id"):
        assert g["cycle"].tolist() == list(range(1, len(g) + 1))
        assert g["rul"].iloc[-1] == 0
        assert np.all(np.diff(g["rul"]) == -1)
    assert set(fleet["failure_mode"]) <= set(FAILURE_MODES)


def test_dropouts_exist_but_never_on_the_first_cycle(fleet):
    assert fleet[list(SENSORS)].isna().to_numpy().any()
    first = fleet[fleet["cycle"] == 1]
    assert not first[list(SENSORS)].isna().to_numpy().any()


def test_failure_modes_show_up_in_the_expected_sensor(fleet):
    def end_minus_start(sensor, mode):
        rows = fleet[fleet["failure_mode"] == mode]
        start = rows[rows["cycle"] <= 20].groupby("machine_id")[sensor].mean()
        end = rows[rows["rul"] <= 5].groupby("machine_id")[sensor].mean()
        return float((end - start).mean())

    assert end_minus_start("vibration", "bearing_wear") > 2 * end_minus_start("vibration", "seal_leak")
    assert end_minus_start("pressure", "seal_leak") < -0.5
    assert end_minus_start("temperature", "overheating") > 8
