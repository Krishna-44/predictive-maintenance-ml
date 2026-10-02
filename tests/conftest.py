import pytest

from predmaint.simulate import SimulationConfig, simulate_fleet
from predmaint.train import train


@pytest.fixture(scope="session")
def fleet():
    return simulate_fleet(SimulationConfig(n_machines=48, seed=11))


@pytest.fixture(scope="session")
def bundle(fleet):
    return train(fleet, with_diagnostics=False)
