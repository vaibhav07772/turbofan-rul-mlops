FROM python:3.11-slim

WORKDIR /app

# System dependencies (libgomp1 for XGBoost/LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app/ ./app/
COPY run.py .

# Trained model (commit via git — small file)
COPY data/models/ ./data/models/

EXPOSE 8000

CMD ["python", "run.py"]