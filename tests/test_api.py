from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GOOD = {
    "students_registered": 600, "attendance_pct": 75, "menu_rating": 3.5,
    "temp_c": 28.0, "cooked_kg": 350.0, "plate_size_cm": 27.0,
    "meal": "lunch", "hostel": "Girls Hostel",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert r.json()["waste_kg"] >= 0


def test_cooked_kg_above_1000_rejected():
    r = client.post("/predict", json={**GOOD, "cooked_kg": 1001})
    assert r.status_code == 422


def test_plate_size_below_10_rejected():
    r = client.post("/predict", json={**GOOD, "plate_size_cm": 9.5})
    assert r.status_code == 422


def test_missing_field_rejected():
    bad = dict(GOOD)
    bad.pop("hostel")
    assert client.post("/predict", json=bad).status_code == 422
