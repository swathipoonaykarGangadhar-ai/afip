FROM python:3.12-slim

WORKDIR /app

# System deps for xgboost / psycopg build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Writable dir for the sqlite checkpoint fallback and trained model
RUN mkdir -p /app/data && chmod 777 /app/data

EXPOSE 8001

# Uvicorn with multiple workers for production. NOTE: the in-memory
# graph store and STATE dict in app/api/main.py are per-process --
# with >1 worker each process gets its OWN synthetic dataset and case
# store. Fine for this prototype/demo deployment; once you swap in
# real Postgres-backed case storage and Neo4j (not in-memory graph),
# switch to multiple workers safely. For now, run with 1 worker.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
