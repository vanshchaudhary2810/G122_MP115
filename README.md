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
pytest
```
