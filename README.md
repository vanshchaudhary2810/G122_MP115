# MP115 — Hostel Mess Food Waste (Bhopal)

Group **G122** · Problem **MP115** · Faculty: Dr. Pulkit Dwivedi · Batch 59

| Enrolment | Name |
|---|---|
| S24CSEU1734 | Vansh Chaudhary |
| S24CSEU1784 | Smarth Bhardwaj |
| S24CSEU2591 | Vansh Choudhary |
| S24CSEU1746 | Aditya Saini |

Predict `waste_kg` (regression, metric MAE).

```bash
pip install -r requirements.txt
dvc pull                              # get data/raw/*.csv
PYTHONPATH=src python -m mp115.checks # find the 7 data problems
PYTHONPATH=src python -m mp115.train  # clean, train v1 with MLflow, register @champion
PYTHONPATH=src python -m mp115.monitor  # PSI drift, per-hostel MAE, retrain v2, promotion rule
PYTHONPATH=src python -m mp115.predict  # holdout predictions with the @champion model
pytest

# service
PYTHONPATH=src uvicorn app.main:app --reload          # http://127.0.0.1:8000/docs
docker build -t mp115-api:1.0 . && docker run -p 8000:8000 mp115-api:1.0
kubectl apply -f k8s/
```

## Layout
| Path | What |
|---|---|
| `src/mp115/checks.py` | finds the seven data problems |
| `src/mp115/clean.py` | cleaning |
| `src/mp115/train.py` | v1 training, MLflow tracking, registry, `@champion` |
| `src/mp115/monitor.py` | PSI, MAE per hostel, v2 retrain, promotion |
| `src/mp115/predict.py` | holdout predictions |
| `app/main.py` | FastAPI `POST /predict`, `GET /health`, JSON logs |
| `Dockerfile`, `k8s/`, `.github/workflows/ci.yml` | container, Kubernetes, CI |
| `reports/findings.yaml` | filled form |

