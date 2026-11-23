import pandas as pd

from mp115.checks import duplicate_rows, spelling_variants, target_typos, unit_mixup
from mp115.clean import clean

P = {
    "target": "waste_kg",
    "numeric": ["students_registered", "attendance_pct", "menu_rating", "temp_c",
                "cooked_kg", "plate_size_cm"],
    "categorical": ["meal", "hostel"],
    "cleaning": {"unit_column": "plate_size_cm", "unit_group": "Boys Hostel 1",
                 "unit_factor": 10, "target_typo_threshold": 100,
                 "stuck_column": "attendance_pct"},
}


def small_df():
    rows = []
    for i, h in enumerate(["Boys Hostel 1", "Boys Hostel 2", "Girls Hostel"] * 4):
        plate = 270.0 if h == "Boys Hostel 1" else 27.0
        rows.append({"record_id": f"R{i}", "date": f"2026-01-{i + 1:02d}",
                     "students_registered": 500, "attendance_pct": 70, "menu_rating": 3.0,
                     "temp_c": 25.0, "cooked_kg": 300.0, "plate_size_cm": plate,
                     "meal": "lunch", "hostel": h, "waste_kg": 40.0})
    return pd.DataFrame(rows)


def test_duplicates_counted_and_removed():
    df = small_df()
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    assert duplicate_rows(df)["count"] == 2
    assert len(clean(df, P)) == 12


def test_unit_mixup_found_and_fixed():
    df = small_df()
    r = unit_mixup(df, P["numeric"])
    assert r["present"] and r["column"] == "plate_size_cm" and r["factor"] == 10
    assert clean(df, P)["plate_size_cm"].max() == 27.0


def test_target_typo_divided_by_ten():
    df = small_df()
    df.loc[0, "waste_kg"] = 400.0
    assert target_typos(df, "waste_kg", 100)["count"] == 1
    assert clean(df, P).loc[0, "waste_kg"] == 40.0


def test_no_spelling_variants_when_clean():
    df = small_df()
    assert spelling_variants(df, ["meal", "hostel"], df)["present"] is False
