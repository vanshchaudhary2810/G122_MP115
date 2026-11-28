# Image for the MP115 prediction API  ->  docker build -t mp115-api:1.0 .
FROM python:3.12-slim

WORKDIR /app

# install libraries first, so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# then copy the code and the champion model
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY params.yaml .

ENV PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
