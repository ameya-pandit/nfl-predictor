FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py .
COPY model_v1_65pct.pt .
COPY scaler_v1_65pct.pkl .

CMD ["python", "pipeline.py"]
