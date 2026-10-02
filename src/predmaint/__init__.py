"""predmaint — predictive maintenance from machine telemetry."""

from predmaint.features import build_features
from predmaint.service import MaintenanceAdvisor, load_or_train
from predmaint.simulate import SimulationConfig, simulate_fleet
from predmaint.train import CostModel, ModelBundle, train

__all__ = [
    "CostModel",
    "MaintenanceAdvisor",
    "ModelBundle",
    "SimulationConfig",
    "build_features",
    "load_or_train",
    "simulate_fleet",
    "train",
]

__version__ = "0.1.0"
