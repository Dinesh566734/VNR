FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# PyG 2.7 can run without the optional compiled extension packages for the
# operations used in this project. Keeping the default image on the minimal
# install path avoids wheel-matrix mismatches between the latest PyTorch release
# and `data.pyg.org` extension indexes during Docker builds.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

COPY config /app/config
COPY src /app/src
COPY data /app/data

RUN mkdir -p /app/models /app/logs \
    && if [ -f /app/data/processed/real_run_150k/sentinel_gat_best.pt ]; then \
        cp /app/data/processed/real_run_150k/sentinel_gat_best.pt /app/models/sentinel_gat_best.pt; \
    fi \
    && if [ -f /app/data/processed/real_run_150k/graph_train_smote.pt ]; then \
        cp /app/data/processed/real_run_150k/graph_train_smote.pt /app/data/processed/graph_train_smote.pt; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "src.inference.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
