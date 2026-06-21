# Slim base keeps the image small; python:3.11 is well-supported by faiss & torch.
FROM python:3.11-slim

# System deps occasionally needed by faiss / scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so Docker layer-caches them independently of code changes.
COPY requirements.txt .
# --timeout and --retries make pip resilient to slow/unstable connections
# (the AI libraries like torch are large and can time out on first download).
RUN pip install --no-cache-dir --timeout 300 --retries 10 -r requirements.txt

# Copy application code and data.
COPY app ./app
COPY data ./data

EXPOSE 8000

# Run as non-root for k8s security context best practice.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Uvicorn serves the FastAPI app. --host 0.0.0.0 so the container is reachable.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
