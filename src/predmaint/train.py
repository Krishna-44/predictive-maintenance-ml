"""Training and evaluation.

Three models are trained on the same causal features:

1. **Failure classifier** — will this machine fail within the next ``horizon`` cycles?
   Gradient-boosted trees, compared against a logistic-regression baseline. The alert threshold is
   chosen to minimise expected maintenance cost (a missed failure costs far more than an
   unnecessary inspection), using out-of-fold predictions on the *training* machines only.
2. **Remaining-useful-life regressor** — how many cycles are left? Trained on RUL capped at
   ``RUL_CAP`` (early in life the exact number is unknowable and irrelevant).
3. **Anomaly detector** — an Isolation Forest fitted only on healthy-period data, useful when a new
   failure mode appears that the classifier has never seen.

All evaluation uses a **machine-level split**: no machine contributes rows to both train and test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predmaint.features import build_features, feature_columns

RUL_CAP = 125
HEALTHY_RUL = 150  # cycles before failure that we treat as "clearly healthy" for the anomaly model


@dataclass(frozen=True)
class CostModel:
    """Relative costs used to pick the alert threshold."""

    missed_failure: float = 10.0  # unplanned breakdown: downtime, emergency repair
    false_alarm: float = 1.0  # an inspection that finds nothing

    def expected_cost(self, y_true: np.ndarray, alerts: np.ndarray) -> float:
        missed = np.sum((y_true == 1) & ~alerts)
        false = np.sum((y_true == 0) & alerts)
        return float(missed * self.missed_failure + false * self.false_alarm) / len(y_true)


def choose_threshold(y_true: np.ndarray, proba: np.ndarray, cost: CostModel) -> float:
    """The probability threshold that minimises expected cost on the given predictions."""
    candidates = np.unique(np.round(np.concatenate([[0.0, 1.0], proba]), 4))
    costs = [cost.expected_cost(y_true, proba >= c) for c in candidates]
    return float(candidates[int(np.argmin(costs))])


def make_classifier(seed: int = 0) -> HistGradientBoostingClassifier:
    # No class re-weighting: probabilities stay interpretable, and the cost-based threshold
    # handles the class imbalance instead.
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0,
        early_stopping=True, n_iter_no_change=15, random_state=seed,
    )


def make_regressor(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=15, early_stopping=True,
        n_iter_no_change=15, random_state=seed,
    )


def make_baseline(seed: int = 0):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))


@dataclass
class ModelBundle:
    classifier: Any
    regressor: Any
    anomaly: Any
    threshold: float
    horizon: int
    features: list[str]
    anomaly_scale: tuple[float, float]  # healthy score mean/std, to report a z-score
    metrics: dict[str, Any] = field(default_factory=dict)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features[self.features]
        proba = self.classifier.predict_proba(X)[:, 1]
        rul = np.clip(self.regressor.predict(X), 0, RUL_CAP)
        raw = -self.anomaly.score_samples(X)
        mu, sd = self.anomaly_scale
        return pd.DataFrame(
            {
                "failure_probability": proba,
                "alert": proba >= self.threshold,
                "rul_estimate": rul,
                "anomaly_z": (raw - mu) / sd,
            },
            index=features.index,
        )


def _classification_metrics(y: np.ndarray, proba: np.ndarray, threshold: float, cost: CostModel) -> dict:
    alerts = proba >= threshold
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "precision": precision_score(y, alerts, zero_division=0),
        "recall": recall_score(y, alerts, zero_division=0),
        "f1": f1_score(y, alerts, zero_division=0),
        "cost_per_cycle": cost.expected_cost(y, alerts),
    }


def _rul_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """RUL error overall (capped target) and in the final 30 cycles, where it matters most."""
    final = true <= 30
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mae_final_30_cycles": float(mean_absolute_error(true[final], pred[final])),
        # Share of late-life predictions that are optimistic (predict failure later than reality).
        "late_share_final_30_cycles": float(np.mean(pred[final] > true[final])),
    }


def _lead_times(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    """Cycles of warning before each machine's failure (-1 if it was never flagged).

    Only alerts in the final ``3 * horizon`` cycles count as warnings; earlier ones are counted as
    false alarms instead of being credited as a very early warning.
    """
    leads = []
    for _, g in frame.groupby("machine_id"):
        late = g[(g["rul"] <= 3 * horizon) & g["alert"]]
        leads.append(int(late["rul"].max()) if len(late) else -1)
    return np.array(leads)


def _summarise_leads(leads: np.ndarray) -> dict:
    caught = leads >= 0
    return {
        "machines": int(len(leads)),
        "failures_flagged": float(caught.mean()),
        "flagged_10_cycles_early": float((leads >= 10).mean()),
        "median_lead_cycles": float(np.median(leads[caught])) if caught.any() else 0.0,
    }


def _machine_level(test: pd.DataFrame, alerts: np.ndarray, horizon: int) -> dict:
    """Operational view: how early is each failure flagged, and how noisy is the alarm?"""
    frame = test.assign(alert=alerts)
    healthy = frame["rul"] > 3 * horizon
    summary = _summarise_leads(_lead_times(frame, horizon))
    summary["false_alarm_rate_healthy"] = float(frame.loc[healthy, "alert"].mean())
    if "failure_mode" in frame:
        summary["by_failure_mode"] = {
            mode: _summarise_leads(_lead_times(group, horizon))
            for mode, group in frame.groupby("failure_mode")
        }
    return summary


def train(
    raw: pd.DataFrame,
    *,
    horizon: int = 20,
    seed: int = 0,
    test_size: float = 0.25,
    cost: CostModel | None = None,
    with_diagnostics: bool = True,
) -> ModelBundle:
    cost = cost or CostModel()
    feats = build_features(raw)
    cols = feature_columns(feats)
    extra = [c for c in ("rul", "failure_mode") if c in raw]
    data = feats.join(raw.sort_values(["machine_id", "cycle"]).reset_index(drop=True)[extra])
    y = (data["rul"] <= horizon).astype(int).to_numpy()
    groups = data["machine_id"].to_numpy()

    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(split.split(data, y, groups))
    X_train, X_test = data.iloc[train_idx][cols], data.iloc[test_idx][cols]
    y_train, y_test = y[train_idx], y[test_idx]

    # Threshold from out-of-fold predictions on training machines (the test set stays untouched).
    oof = np.zeros(len(train_idx))
    for fit_idx, val_idx in GroupKFold(n_splits=5).split(X_train, y_train, groups[train_idx]):
        model = make_classifier(seed).fit(X_train.iloc[fit_idx], y_train[fit_idx])
        oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
    threshold = choose_threshold(y_train, oof, cost)

    clf = make_classifier(seed).fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]

    rul_train = np.minimum(data["rul"].to_numpy()[train_idx], RUL_CAP)
    rul_test = np.minimum(data["rul"].to_numpy()[test_idx], RUL_CAP)
    reg = make_regressor(seed).fit(X_train, rul_train)
    rul_pred = np.clip(reg.predict(X_test), 0, RUL_CAP)

    healthy = data.iloc[train_idx]["rul"].to_numpy() > HEALTHY_RUL
    iso = IsolationForest(n_estimators=300, random_state=seed).fit(X_train[healthy])
    healthy_scores = -iso.score_samples(X_train[healthy])
    scale = (float(healthy_scores.mean()), float(healthy_scores.std() or 1.0))

    bundle = ModelBundle(clf, reg, iso, threshold, horizon, cols, scale)
    alerts = proba >= threshold
    metrics: dict[str, Any] = {
        "data": {
            "machines": int(len(np.unique(groups))),
            "rows": int(len(data)),
            "train_machines": int(len(np.unique(groups[train_idx]))),
            "test_machines": int(len(np.unique(groups[test_idx]))),
            "positive_rate": float(y.mean()),
            "horizon_cycles": horizon,
        },
        "threshold": threshold,
        "cost_model": {"missed_failure": cost.missed_failure, "false_alarm": cost.false_alarm},
        "classifier": _classification_metrics(y_test, proba, threshold, cost),
        "machine_level": _machine_level(data.iloc[test_idx], alerts, horizon),
        "rul": _rul_metrics(rul_test, rul_pred),
        "anomaly": {"roc_auc": float(roc_auc_score(y_test, -iso.score_samples(X_test)))},
    }

    if with_diagnostics:
        # The alert threshold is a business decision: show the trade-off across cost ratios,
        # each threshold chosen on training folds and evaluated on the held-out machines.
        sweep = []
        for ratio in (2, 5, 10, 20, 50):
            c = CostModel(missed_failure=ratio, false_alarm=1.0)
            th = choose_threshold(y_train, oof, c)
            ml = _machine_level(data.iloc[test_idx], proba >= th, horizon)
            sweep.append({
                "miss_to_false_alarm_cost": ratio,
                "threshold": th,
                "recall": recall_score(y_test, proba >= th, zero_division=0),
                "precision": precision_score(y_test, proba >= th, zero_division=0),
                "flagged_10_cycles_early": ml["flagged_10_cycles_early"],
                "median_lead_cycles": ml["median_lead_cycles"],
                "false_alarm_rate_healthy": ml["false_alarm_rate_healthy"],
            })
        metrics["cost_sweep"] = sweep

        # Baselines: what the model has to beat.
        base = make_baseline(seed).fit(X_train, y_train)
        base_proba = base.predict_proba(X_test)[:, 1]
        never = np.zeros_like(y_test, dtype=bool)
        metrics["baselines"] = {
            "logistic_regression": {
                "roc_auc": roc_auc_score(y_test, base_proba),
                "pr_auc": average_precision_score(y_test, base_proba),
            },
            "never_alert_cost_per_cycle": cost.expected_cost(y_test, never),
        }
        # Leakage demo: a random row split puts neighbouring cycles of the same machine on both
        # sides, so the model can "recognise" machines instead of learning degradation.
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            data[cols], y, test_size=test_size, random_state=seed, stratify=y
        )
        leaky = make_classifier(seed).fit(Xr_tr, yr_tr).predict_proba(Xr_te)[:, 1]
        metrics["leakage_demo"] = {
            "random_row_split_pr_auc": average_precision_score(yr_te, leaky),
            "machine_split_pr_auc": metrics["classifier"]["pr_auc"],
        }
        sample = np.random.default_rng(seed).choice(len(X_test), size=min(4000, len(X_test)), replace=False)
        imp = permutation_importance(
            clf, X_test.iloc[sample], y_test[sample], scoring="average_precision",
            n_repeats=5, random_state=seed,
        )
        order = np.argsort(-imp.importances_mean)
        metrics["top_features"] = [
            {"feature": cols[i], "importance": float(imp.importances_mean[i])} for i in order[:10]
        ]
        distractors = [i for i, c in enumerate(cols) if c.startswith(("humidity", "voltage"))]
        metrics["distractor_importance_max"] = float(imp.importances_mean[distractors].max())

    bundle.metrics = _rounded(metrics)
    return bundle


def _rounded(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 4)
    return obj
