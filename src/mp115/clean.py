"""Turn the messy train.csv into a clean training table."""
import numpy as np
import pandas as pd


def clean(df, p, stuck=None):
    """stuck: dict from checks.stuck_sensor (start_date, end_date, column) or None."""
    c = p["cleaning"]
    df = df.copy()
    df = df.drop_duplicates()
    for col in p["categorical"]:                        # tidy text
        df[col] = df[col].astype(str).str.strip()
    # unit mix-up: one hostel typed plate size in mm, not cm
    mask = df["hostel"] == c["unit_group"]
    df.loc[mask, c["unit_column"]] = df.loc[mask, c["unit_column"]] / c["unit_factor"]
    # target typed 10x too big
    big = df[p["target"]] > c["target_typo_threshold"]
    df.loc[big, p["target"]] = df.loc[big, p["target"]] / 10
    # stuck sensor: the readings in that window are not real -> blank them, then fill
    if stuck and stuck.get("present"):
        win = df["date"].between(stuck["start_date"], stuck["end_date"]) & \
              (df[stuck["column"]] == stuck["value"])
        df.loc[win, stuck["column"]] = np.nan
    for col in p["numeric"]:
        df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)


def features(df, p):
    return df[p["numeric"] + p["categorical"]]
