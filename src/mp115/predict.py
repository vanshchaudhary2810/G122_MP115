"""Predict waste_kg for holdout_features.csv with the @champion model.

Run:  python -m mp115.predict
"""
import os

import joblib
import pandas as pd

from mp115.clean import features
from mp115.config import load_params


def main():
    p = load_params()
    model = joblib.load(p["predict"]["model_path"])     # copy of the @champion version
    df = pd.read_csv(p["data"]["holdout"])
    for col in p["categorical"]:
        df[col] = df[col].astype(str).str.strip()
    pred = model.predict(features(df, p)).clip(min=0)
    out = pd.DataFrame({"record_id": df["record_id"], "prediction": pred.round(3)})
    os.makedirs(os.path.dirname(p["predict"]["output"]), exist_ok=True)
    out.to_csv(p["predict"]["output"], index=False)
    print(f"wrote {len(out)} rows to {p['predict']['output']}")


if __name__ == "__main__":
    main()
