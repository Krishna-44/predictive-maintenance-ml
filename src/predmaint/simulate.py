"""Synthetic run-to-failure telemetry for a fleet of industrial machines.

Each machine runs from commissioning until it fails. Health degrades slowly at first and then
faster (``damage = progress ** p``), and the dominant failure mode decides which sensors show it:

* ``bearing_wear``     → vibration rises, rpm sags and torque creeps up (friction)
* ``overheating``      → temperature rises
* ``seal_leak``        → hydraulic pressure drops
* ``electrical_fault`` → motor current rises and gets noisy, with only a short warning period

On top of that the simulator adds what makes real telemetry awkward: a per-cycle operating setting
(``duty``) that moves temperature, torque, current and vibration on its own, machine-to-machine
baseline differences, measurement noise, transient vibration glitches, sensor dropouts (missing
values) and two sensors (``humidity``, ``voltage``) that carry no information about failure.

The data is synthetic so the project is fully reproducible; the modelling code only assumes the
column layout, so it runs unchanged on real telemetry with the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

INFORMATIVE_SENSORS = ("temperature", "vibration", "pressure", "rpm", "torque", "current")
DISTRACTOR_SENSORS = ("humidity", "voltage")
SENSORS = INFORMATIVE_SENSORS + DISTRACTOR_SENSORS
SIGNALS = ("duty", *SENSORS)  # every numeric reading, including the operating setting

LOAD_FACTORS = {"low": 0.6, "medium": 0.8, "high": 1.0}
FAILURE_MODES = ("bearing_wear", "overheating", "seal_leak", "electrical_fault")
_MODE_PROBS = (0.3, 0.3, 0.3, 0.1)

# Degradation signature per mode: how strongly damage shows in each sensor.
_SIGNATURE = {
    #                  temp  vib   pres  rpm   torque current
    "bearing_wear":     (0.25, 1.0, 0.1,  1.0,  1.0,  0.1),
    "overheating":      (1.0,  0.2, 0.1,  0.2,  0.1,  0.1),
    "seal_leak":        (0.2,  0.2, 1.0,  0.2,  0.1,  0.1),
    "electrical_fault": (0.3,  0.1, 0.05, 0.1,  0.1,  1.0),
}


@dataclass(frozen=True)
class SimulationConfig:
    n_machines: int = 160
    seed: int = 7
    min_life: int = 120
    max_life: int = 360
    dropout_rate: float = 0.005  # share of sensor readings that are missing
    glitch_rate: float = 0.01  # share of cycles with a transient vibration spike


def _simulate_machine(machine_id: int, cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    load_name = str(rng.choice(list(LOAD_FACTORS), p=[0.3, 0.4, 0.3]))
    load = LOAD_FACTORS[load_name]
    mode = str(rng.choice(FAILURE_MODES, p=_MODE_PROBS))

    # Heavier load → shorter life, with plenty of spread between individual machines.
    mean_life = 300 - 130 * (load - 0.6) / 0.4
    life = int(np.clip(rng.normal(mean_life, 45), cfg.min_life, cfg.max_life))
    t = np.arange(1, life + 1)

    # Degradation starts part-way through life; electrical faults develop late and fast.
    onset = rng.uniform(0.85, 0.93) if mode == "electrical_fault" else rng.uniform(0.25, 0.55)
    progress = np.clip((t / life - onset) / (1 - onset), 0, 1)
    damage = progress ** rng.uniform(1.6, 3.0)
    k_temp, k_vib, k_pres, k_rpm, k_torque, k_cur = _SIGNATURE[mode]

    # Operating setting: a bounded random walk around 1.0 that changes every cycle.
    duty = np.clip(1 + np.cumsum(rng.normal(0, 0.02, life)), 0.8, 1.2)
    duty = 0.5 * duty + 0.5 * rng.uniform(0.85, 1.15, life)

    # Manufacturing variation: every machine has its own healthy baseline.
    temp0 = 60 + 15 * load + rng.normal(0, 2.0)
    vib0 = 2.0 + 1.5 * load + rng.normal(0, 0.25)
    pres0 = 5.0 + rng.normal(0, 0.2)
    rpm0 = 1500 * (0.7 + 0.3 * load) + rng.normal(0, 15)
    torque0 = 40 * load + rng.normal(0, 1.5)

    def noise(scale):
        return rng.normal(0, scale, life)

    temperature = temp0 + 12 * (duty - 1) + 14 * k_temp * damage + noise(1.2)
    vibration = vib0 + 1.0 * (duty - 1) + 3.0 * k_vib * damage**1.3 + np.abs(noise(0.25))
    glitches = rng.random(life) < cfg.glitch_rate
    vibration[glitches] += rng.uniform(2, 5, glitches.sum())
    pressure = pres0 - 1.0 * k_pres * damage + noise(0.09)
    rpm = rpm0 * (0.97 + 0.03 * duty) - 35 * k_rpm * damage + noise(12)
    torque = torque0 * duty * (1 + 0.1 * k_torque * damage) + noise(1.0)
    current = 2 + 0.25 * torque + 3.0 * k_cur * damage + noise(0.3 + 0.8 * k_cur * damage)
    humidity = 45 + 10 * np.sin(2 * np.pi * t / rng.uniform(40, 90) + rng.uniform(0, 6.3)) + noise(2)
    voltage = 400 + noise(3)

    frame = pd.DataFrame(
        {
            "machine_id": machine_id,
            "cycle": t,
            "load": load_name,
            "duty": duty,
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "rpm": rpm,
            "torque": torque,
            "current": current,
            "humidity": humidity,
            "voltage": voltage,
            "rul": life - t,  # remaining useful life in cycles; 0 on the cycle it fails
            "failure_mode": mode,
        }
    )
    # Sensor dropouts, but never in the first cycle so every series has a starting value.
    for sensor in SENSORS:
        missing = rng.random(life) < cfg.dropout_rate
        missing[0] = False
        frame.loc[missing, sensor] = np.nan
    return frame


def simulate_fleet(cfg: SimulationConfig | None = None) -> pd.DataFrame:
    """Simulate ``cfg.n_machines`` complete run-to-failure histories (deterministic per seed)."""
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    frames = [_simulate_machine(m, cfg, rng) for m in range(1, cfg.n_machines + 1)]
    return pd.concat(frames, ignore_index=True)
