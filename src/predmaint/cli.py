"""Command-line interface: ``predmaint simulate | train | predict | serve``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from predmaint.service import DEFAULT_MODEL_PATH, MaintenanceAdvisor, load_bundle, save_bundle
from predmaint.simulate import SimulationConfig, simulate_fleet
from predmaint.train import CostModel, train


def cmd_simulate(args: argparse.Namespace) -> int:
    fleet = simulate_fleet(SimulationConfig(n_machines=args.machines, seed=args.seed))
    fleet.to_csv(args.out, index=False)
    print(f"wrote {len(fleet):,} rows for {args.machines} machines to {args.out}")
    return 0


def _print_report(m: dict) -> None:
    c, ml, r = m["classifier"], m["machine_level"], m["rul"]
    d = m["data"]
    print(f"\n{d['machines']} machines ({d['train_machines']} train / {d['test_machines']} test), "
          f"{d['rows']:,} cycles, failure horizon {d['horizon_cycles']} cycles")
    print(f"alert threshold {m['threshold']:.3f} (missed failure : false alarm cost = "
          f"{m['cost_model']['missed_failure']:g} : {m['cost_model']['false_alarm']:g})\n")
    print(f"  ROC-AUC {c['roc_auc']:.3f}   PR-AUC {c['pr_auc']:.3f}   "
          f"precision {c['precision']:.3f}   recall {c['recall']:.3f}")
    print(f"  failures flagged ≥10 cycles early: {ml['flagged_10_cycles_early']:.1%}   "
          f"median warning: {ml['median_lead_cycles']:.0f} cycles   "
          f"false alarms on healthy cycles: {ml['false_alarm_rate_healthy']:.2%}")
    print(f"  RUL MAE {r['mae']:.1f} cycles (final 30 cycles: {r['mae_final_30_cycles']:.1f})   "
          f"anomaly ROC-AUC {m['anomaly']['roc_auc']:.3f}")
    if "leakage_demo" in m:
        lk = m["leakage_demo"]
        print(f"  leakage check: PR-AUC {lk['random_row_split_pr_auc']:.3f} with a random row split "
              f"vs {lk['machine_split_pr_auc']:.3f} with a machine split")


def cmd_train(args: argparse.Namespace) -> int:
    raw = pd.read_csv(args.data) if args.data else simulate_fleet()
    bundle = train(raw, horizon=args.horizon, cost=CostModel(missed_failure=args.miss_cost))
    save_bundle(bundle, args.model)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(bundle.metrics, indent=2) + "\n")
    _print_report(bundle.metrics)
    print(f"\nmodel → {args.model}\nmetrics → {args.report}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    advisor = MaintenanceAdvisor(load_bundle(args.model))
    for result in advisor.assess_fleet(pd.read_csv(args.data)):
        drivers = ", ".join(f"{d['sensor']} {d['z']:+.1f}σ" for d in result["drivers"])
        print(f"machine {result['machine_id']:>4}  cycle {result['cycle']:>4}  "
              f"{result['status'].upper():<5}  p(fail ≤ horizon)={result['failure_probability']:.2f}  "
              f"RUL≈{result['rul_estimate']:.0f}  [{drivers}]")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    os.environ["PREDMAINT_MODEL"] = str(args.model)
    uvicorn.run("predmaint.api:create_app", factory=True, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="predmaint", description="Predictive maintenance toolkit.")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="write a synthetic run-to-failure fleet to CSV")
    sim.add_argument("--machines", type=int, default=160)
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--out", default="fleet.csv")
    sim.set_defaults(func=cmd_simulate)

    tr = sub.add_parser("train", help="train, evaluate and save the models")
    tr.add_argument("--data", help="CSV of run-to-failure histories (default: simulate one)")
    tr.add_argument("--horizon", type=int, default=20, help="predict failure within this many cycles")
    tr.add_argument("--miss-cost", type=float, default=10.0, help="cost of a missed failure vs a false alarm")
    tr.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    tr.add_argument("--report", default="reports/metrics.json")
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="assess the latest cycle of every machine in a CSV")
    pr.add_argument("--data", required=True)
    pr.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    pr.set_defaults(func=cmd_predict)

    sv = sub.add_parser("serve", help="run the API and fleet dashboard")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    sv.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
