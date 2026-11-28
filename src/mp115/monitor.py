"""Stage 4: drift (PSI), per-hostel MAE, retrain v2, promotion rule.

Run:  python -m mp115.monitor
"""
import hashlib
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from mp115.checks import stuck_sensor
from mp115.clean import clean, features
from mp115.config import load_params


def psi(expected, actual, bins=10):
    """Population Stability Index. Bin edges = deciles of the training column."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    p = load_params()
    m, mon, tgt = p["mlflow"], p["monitoring"], p["target"]
    sha = hashlib.sha256(open(p["data"]["train"], "rb").read()).hexdigest()

    raw = pd.read_csv(p["data"]["train"])
    train = clean(raw, p, stuck_sensor(raw, p["numeric"]))
    later = pd.read_csv(p["data"]["later"])

    # 1. drift: PSI for every numeric column
    psis = {c: round(psi(train[c], later[c], mon["psi_bins"]), 4) for c in p["numeric"]}
    most_drifted = max(psis, key=psis.get)

    # 2. v1 (current champion) on later.csv, per hostel
    v1 = joblib.load(p["predict"]["model_path"])
    joblib.dump(v1, "models/model_v1.joblib")
    later["pred_v1"] = v1.predict(features(later, p))
    mae_by_group = {h: round(mean_absolute_error(g[tgt], g["pred_v1"]), 4)
                    for h, g in later.groupby("hostel")}
    worst = max(mae_by_group, key=mae_by_group.get)

    # 3. retrain v2 on train + first 70% of later (by date), compare on last 30%
    later = later.sort_values(["date", "record_id"]).reset_index(drop=True)
    cut = int(len(later) * mon["retrain_fraction"])
    first, check = later.iloc[:cut], later.iloc[cut:]
    both = pd.concat([train, first], ignore_index=True)
    v2 = clone(v1).fit(features(both, p), both[tgt])
    mae_v1 = round(mean_absolute_error(check[tgt], v1.predict(features(check, p))), 4)
    mae_v2 = round(mean_absolute_error(check[tgt], v2.predict(features(check, p))), 4)

    # 4. promotion rule: v2 must be at least 5% better
    promote = mae_v2 <= mae_v1 * (1 - mon["promote_if_better_by"])
    decision = "promote" if promote else "keep"

    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment_name"])
    with mlflow.start_run(run_name="v2_retrain") as run:
        mlflow.log_param("data_sha256", sha)
        mlflow.log_param("trained_on", "train_clean + first 70% of later")
        mlflow.log_param("decision", decision)
        mlflow.log_metric("mae_v1_check", mae_v1)
        mlflow.log_metric("mae_v2_check", mae_v2)
        for c, v in psis.items():
            mlflow.log_metric(f"psi_{c}", v)
        for h, v in mae_by_group.items():
            mlflow.log_metric(f"mae_later_v1_{h.replace(' ', '_')}", v)
        mlflow.sklearn.log_model(v2, name="model", serialization_format="cloudpickle")
        v2_run = run.info.run_id

    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{v2_run}/model", m["registered_model_name"])
    champion = int(mv.version) if promote else 1
    client.set_registered_model_alias(m["registered_model_name"], "champion", str(champion))
    if promote:
        joblib.dump(v2, p["predict"]["model_path"])     # the service loads the champion

    report = {"psi": psis, "most_drifted_feature": most_drifted,
              "mae_by_group": mae_by_group, "worst_served_group": worst,
              "mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2,
              "decision": decision, "v2_run_id": v2_run,
              "v2_version": int(mv.version), "champion_version": champion}
    with open(mon["report"], "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
