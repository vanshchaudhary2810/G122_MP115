"""Detect the seven kinds of data problem in train.csv.

Run:  python -m mp115.checks
"""
import pandas as pd

from mp115.config import load_params

SENTINELS = [-1, -99, -999, 999, 9999, -9999]


def leaking_column(train, holdout, target):
    # a feature that the holdout file does not have was only known after the meal
    extra = [c for c in train.columns if c not in holdout.columns and c != target]
    return {"present": bool(extra), "column": extra[0] if extra else None}


def unit_mixup(df, numeric, group_col="hostel"):
    med = df.groupby(df[group_col].str.strip())[numeric].median()
    for col in numeric:
        for g in med.index:
            others = med[col].drop(g).median()
            if others > 0:
                ratio = med.loc[g, col] / others
                for factor in (10, 12, 60, 100, 1000):
                    if abs(ratio - factor) / factor < 0.25:
                        return {"present": True, "column": col, "group_value": g,
                                "factor": factor, "ratio": round(float(ratio), 2)}
    return {"present": False}


def sentinel_value(df, numeric, min_count=5):
    # a real sentinel repeats many times; 999 twice in a 0-1200 column is a real count
    for col in numeric:
        for v in SENTINELS:
            n = int((df[col] == v).sum())
            if n >= min_count:
                return {"present": True, "column": col, "value": v, "count": n}
    return {"present": False}


def duplicate_rows(df):
    n = int(df.duplicated().sum())
    return {"present": n > 0, "count": n}


def stuck_sensor(df, numeric, min_days=14):
    d = df.drop_duplicates().sort_values("date").reset_index(drop=True)
    best = None
    for col in numeric:
        same = d[col].ne(d[col].shift()).cumsum()        # id of each run of equal values
        for _, run in d.groupby(same):
            days = pd.to_datetime(run["date"])
            span = (days.max() - days.min()).days
            if len(run) > 5 and span >= min_days and (best is None or len(run) > best["rows"]):
                best = {"column": col, "value": run[col].iloc[0], "rows": len(run),
                        "start_date": run["date"].min(), "end_date": run["date"].max()}
    return {"present": True, **best} if best else {"present": False}


def target_typos(df, target, threshold):
    n = int((df[target] > threshold).sum())
    return {"present": n > 0, "count": n}


def spelling_variants(df, cat_cols, reference):
    for col in cat_cols:
        raw = set(df[col].dropna().unique())
        clean = set(reference[col].dropna().unique())
        if raw - clean:
            return {"present": True, "column": col, "true_levels": len(clean),
                    "variants": sorted(raw - clean)}
    return {"present": False}


def run_all(p):
    train = pd.read_csv(p["data"]["train"])
    later = pd.read_csv(p["data"]["later"])
    holdout = pd.read_csv(p["data"]["holdout"])
    num, cat, tgt = p["numeric"], p["categorical"], p["target"]
    return {
        "leaking_column": leaking_column(train, holdout, tgt),
        "unit_mixup": unit_mixup(train, num),
        "sentinel_value": sentinel_value(train, num + [tgt]),
        "duplicate_rows": duplicate_rows(train),
        "stuck_sensor": stuck_sensor(train, num),
        "target_typos": target_typos(train, tgt, p["cleaning"]["target_typo_threshold"]),
        "spelling_variants": spelling_variants(train, cat, later),
    }


if __name__ == "__main__":
    for k, v in run_all(load_params()).items():
        print(f"{k:18s} {v}")
