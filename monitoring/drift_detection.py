import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def simulate_production_data(X_test, drift=True):
    prod = X_test.copy()
    if drift:
        prod["tenure"] = prod["tenure"] * 0.7
        prod["MonthlyCharges"] = prod["MonthlyCharges"] * 1.3
        prod["TotalCharges"] = prod["TotalCharges"] * 1.2
    return prod


def run_drift_report(reference_data, current_data):
    logger.info("Running data drift report...")
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data)

    os.makedirs("monitoring", exist_ok=True)

    # Save full result as JSON
    result = snapshot.dict()
    with open("monitoring/drift_report.json", "w") as f:
        json.dump(result, f, default=str)

    # Extract drift summary
    metrics = result.get("metrics", [])
    drift_detected = False
    drift_share = 0.0
    for m in metrics:
        r = m.get("result", {})
        if "share_of_drifted_columns" in r:
            drift_share = r["share_of_drifted_columns"]
            drift_detected = r.get("dataset_drift", drift_share > 0.5)
            break

    summary = {
        "drift_detected": bool(drift_detected),
        "drift_share": round(float(drift_share), 3),
        "drifted_columns": int(drift_share * len(reference_data.columns))
    }
    with open("monitoring/drift_summary.json", "w") as f:
        json.dump(summary, f)

    logger.info(f"Drift detected: {drift_detected} | Drift share: {drift_share:.1%}")
    return summary


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    prod_data = simulate_production_data(X_test, drift=True)
    summary = run_drift_report(X_train, prod_data)
    print(f"\n✅ Drift Report Generated!")
    print(f"Drift detected: {summary['drift_detected']}")
    print(f"Drifted columns: {summary['drifted_columns']}")