# Sentinel-UPI

Sentinel-UPI is a graph-based fraud detection system for UPI-style transactions built with PyTorch Geometric, FastAPI, and a React/Recharts dashboard.

The repository currently includes:

- PaySim-to-UPI data adaptation
- Heterogeneous graph construction and feature engineering
- Graph-SMOTE augmentation and graph partitioning
- Sentinel-GAT training, evaluation, explainability, and inference API
- Docker/Kubernetes deployment assets
- Analytics dashboard backend and frontend scaffold
- Baseline benchmark pipeline

## Current Status

The project runs end-to-end and the full test suite passes.

```powershell
python -m pytest tests/ -v --tb=short
```

The real-data training and benchmark pipeline has also been run against `data/raw/paysim.csv` using a validated `150000`-row slice. The current implementation is functional, but real-data model quality is still below the target metrics from the implementation plan and needs further debugging.

## Project Layout

```text
sentinel-upi/
├── config/
├── data/
│   ├── raw/
│   └── processed/
├── docker/
├── k8s/
├── results/
├── src/
│   ├── dashboard/
│   ├── data/
│   ├── inference/
│   ├── models/
│   └── training/
└── tests/
```

## Requirements

- Python 3.11
- PowerShell on Windows for the commands below
- Node.js 18+ for the dashboard frontend
- Docker Desktop if you want to use the container stack

## Setup

From the project root:

```powershell
cd "c:\Users\Dinesh Kumar\Desktop\VNR\Project\sentinel-upi"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset

Expected raw dataset path:

```text
data/raw/paysim.csv
```

This repository already has a real PaySim file in that location.

## Run Tests

```powershell
python -m pytest tests/ -v --tb=short
```

## Train on Real Data

The current graph feature pipeline is heavy for the full PaySim corpus, so the validated run uses a bounded slice:

```powershell
python -m src.training.train --csv-path data/raw/paysim.csv --output-dir data/processed/real_run_150k --max-rows 150000 --device cpu --max-epochs 8 --patience 4
```

Artifacts from that run are written to:

```text
data/processed/real_run_150k/
```

Key files:

- `graph_train.pt`
- `graph_val.pt`
- `graph_test.pt`
- `graph_train_smote.pt`
- `cluster_assignments.pkl`
- `sentinel_gat_best.pt`
- `training_summary.json`
- `test_evaluation.json`

## Run Benchmarks

Generate the Phase 10 baseline benchmark report:

```powershell
@'
from pathlib import Path
from src.training.ablation import run_benchmarks_from_processed_artifacts

report = run_benchmarks_from_processed_artifacts(
    processed_dir=Path(r"data/processed/real_run_150k"),
    output_path=Path(r"results/benchmark.md"),
    device="cpu",
    verbose=True,
)
print(report.output_path)
'@ | python -
```

Outputs:

- `results/benchmark.md`
- `results/benchmark.json`

## Run the Inference API

The default API bootstrap looks for graph artifacts directly under `data/processed/`, so copy the trained graph artifact once before starting the server:

```powershell
Copy-Item data\processed\real_run_150k\graph_train_smote.pt data\processed\graph_train_smote.pt -Force
python -m uvicorn src.inference.api:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

Health check:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

Example score request:

```powershell
$headers = @{ "X-API-Key" = "dev-secret-key" }
$body = @{
  txn_id = "TXN_LOCAL_001"
  src_upi = "user123@okicici"
  dst_upi = "merchant456@oksbi"
  amount = 2500.0
  timestamp = "2024-03-15T14:23:11Z"
  merchant_type = "gaming_wallet"
  device_id = "device_abc"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/score -Method Post -Headers $headers -Body $body -ContentType "application/json"
```

## Run the Dashboard

### Backend

```powershell
python -m uvicorn src.dashboard.backend:app --host 0.0.0.0 --port 8001 --reload
```

The dashboard backend automatically loads the latest processed run, with `data/processed/real_run_150k` preferred when present.

### Frontend

Open a second terminal:

```powershell
cd "c:\Users\Dinesh Kumar\Desktop\VNR\Project\sentinel-upi\src\dashboard\frontend"
npm install
$env:VITE_DASHBOARD_API_BASE="http://127.0.0.1:8001"
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Run with Docker

From the repo root:

```powershell
docker compose -f docker/docker-compose.yml up --build
```

Services exposed by the compose stack:

- API: `http://127.0.0.1:8000`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`
- RedisGraph: `127.0.0.1:6379`

## Kubernetes Assets

The repository includes:

- `k8s/deployment.yaml`
- `k8s/hpa.yaml`

These define a 3-replica API deployment, health probes, resource limits, GPU node selection, and autoscaling rules.

## Notes

- The implementation plan is functionally covered through the benchmark phase.
- The biggest remaining task is model-quality debugging on real data, not additional scaffolding.
- If you retrain into a different processed run directory, update the API and benchmark commands accordingly.
