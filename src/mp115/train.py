"""Train v1 on cleaned train.csv with MLflow tracking, register best as @champion.

Run:  python -m mp115.train
"""
import hashlib

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mp115.checks import stuck_sensor
from mp115.clean import clean, features
from mp115.config import load_params


def make_model(cfg, p):
    cfg = dict(cfg)
    kind = cfg.pop("model")
    reg = {"ridge": Ridge, "random_forest": RandomForestRegressor,
           "gradient_boosting": GradientBoostingRegressor}[kind]
    if kind != "ridge":
        cfg["random_state"] = p["mlflow"]["random_state"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), p["numeric"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), p["categorical"]),
    ])
    return Pipeline([("pre", pre), ("reg", reg(**cfg))])


def main():
    p = load_params()
    m = p["mlflow"]
    sha = hashlib.sha256(open(p["data"]["train"], "rb").read()).hexdigest()
    raw = pd.read_csv(p["data"]["train"])
    df = clean(raw, p, stuck_sensor(raw, p["numeric"]))
    X, y = features(df, p), df[p["target"]]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=m["test_size"],
                                          random_state=m["random_state"])

    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment_name"])
    best = None
    for cfg in m["runs"]:
        with mlflow.start_run(run_name=f"v1_{cfg['model']}") as run:
            model = make_model(cfg, p).fit(Xtr, ytr)
            mae = mean_absolute_error(yva, model.predict(Xva))
            mlflow.log_param("data_sha256", sha)
            mlflow.log_params({k: str(v) for k, v in cfg.items()})
            mlflow.log_param("rows_clean", len(df))
            mlflow.log_metric("mae_val", mae)
            mlflow.sklearn.log_model(model, name="model",
                                     serialization_format="cloudpickle")
            print(f"{run.info.run_id}  {cfg}  mae_val={mae:.3f}")
            if best is None or mae < best[1]:
                best = (run.info.run_id, mae, model)

    run_id, mae, model = best
    mv = mlflow.register_model(f"runs:/{run_id}/model", m["registered_model_name"])
    MlflowClient().set_registered_model_alias(m["registered_model_name"], "champion", mv.version)
    joblib.dump(model, "models/model.joblib")
    print(f"best run {run_id} mae={mae:.3f} -> {m['registered_model_name']} v{mv.version} @champion")


if __name__ == "__main__":
    main()
