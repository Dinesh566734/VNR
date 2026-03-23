You are building Sentinel-UPI — a real-time Graph Attention Network (GAT) 
fraud detection engine for UPI transactions, based on the Aggarwal et al. 
(IJRTMR 2026) paper. This is a complete implementation from scratch.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Target metrics:
- F1-score ≥ 0.964
- AUC-ROC ≥ 0.98
- Inference latency < 20ms per transaction
- Must beat XGBoost F1 (0.893) by > 7%

Stack:
- Python 3.11
- PyTorch 2.x + PyTorch Geometric (PyG)
- Redis Graph (in-memory graph DB)
- FastAPI (inference API)
- Docker + Kubernetes (serving)
- React + Recharts (analytics dashboard)
- AWS SNS (real-time alerting)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create this exact structure:

sentinel-upi/
├── CLAUDE.md                    # project memory file
├── config/
│   └── hparams.yaml             # all hyperparameters (never hardcode)
├── data/
│   ├── raw/                     # original PaySim CSV
│   └── processed/               # graph .pt files + cluster assignments
├── src/
│   ├── data/
│   │   ├── paysim_loader.py     # load + UPI adaptation
│   │   ├── graph_builder.py     # build HeteroData graph
│   │   ├── feature_engineer.py  # node + edge features
│   │   ├── smote.py             # Graph-SMOTE implementation
│   │   └── partitioner.py       # METIS mini-batch partitioning
│   ├── models/
│   │   ├── gat.py               # Sentinel-GAT architecture
│   │   ├── focal_loss.py        # Focal Loss implementation
│   │   └── explainer.py         # GNN Explainer (XAI layer)
│   ├── training/
│   │   ├── train.py             # main training loop
│   │   ├── eval.py              # evaluation + confusion matrix
│   │   └── ablation.py          # ablation studies
│   ├── inference/
│   │   ├── api.py               # FastAPI inference server
│   │   ├── graph_cache.py       # Redis subgraph retrieval
│   │   └── alerting.py          # AWS SNS integration
│   └── dashboard/
│       ├── backend.py           # dashboard data endpoints
│       └── frontend/            # React + Recharts app
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── k8s/
│   ├── deployment.yaml
│   └── hpa.yaml
└── requirements.txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — DATA PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: src/data/paysim_loader.py

Load PaySim CSV and apply three UPI-specific adaptations:

1. NPCI regulatory cap
   - Clip all P2P transactions to ₹100,000 maximum
   - Transactions exceeding this must be split across multiple VPAs
   - This simulates "smurfing" — the exact fraud pattern we're catching

2. MCC risk weighting
   - Assign merchant_risk_weight to each merchant node:
     * Utility / government portals: weight = 0.5 (low risk)
     * Standard retail (Zomato, Amazon, etc.): weight = 1.0
     * Unregulated gaming wallets: weight = 1.5
     * Crypto exchanges / offshore: weight = 2.0
   - This weight becomes a baseline edge feature entering the GAT

3. Temporal velocity injection
   - Modulate transaction generation with a sinusoidal pattern:
     * Peak hours: 13:00–14:00 and 18:00–20:00 (legitimate surge)
     * Off-hours: 01:00–04:00 (fraud burst — automated scripts)
   - The model must learn to distinguish malicious 3AM bursts
     from organic lunch-hour spikes

Output: cleaned DataFrame with columns:
  txn_id, src_upi, dst_upi, amount_clipped, timestamp,
  merchant_type, mcc_weight, is_fraud (label)

Validation assertions:
  - Fraud rate must be between 0.10% and 0.20%
  - No transaction amount exceeds ₹100,000
  - Timestamp range spans exactly 30 days

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — GRAPH CONSTRUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: src/data/graph_builder.py + feature_engineer.py

Model the UPI network as a directed multigraph G = (V, E) using
torch_geometric.data.HeteroData.

NODE FEATURES (per user/merchant node):
  1. account_age          — days since account creation (normalized 0–1)
  2. kyc_tier             — 0=unverified, 1=basic, 2=full KYC
  3. daily_avg_spend      — rolling 30-day average (log-normalized)
  4. pagerank_local       — localized PageRank score
                           (spike = account taken over as pass-through)
  5. betweenness_central  — betweenness centrality
                           (sudden spike = money mule hub)

EDGE FEATURES (per transaction):
  1. log_amount           — log(amount + 1) to normalize ₹1–₹100,000 range
  2. delta_t              — seconds since this user's previous transaction
  3. temporal_decay       — w_ij = e^(−0.01 × delta_t)
                           (recent transactions weighted heavier)
  4. dfs_cycle_flag       — binary: 1 if this edge completes a cycle
                           (run lightweight DFS, A→B→C→A = money laundering)
  5. mcc_weight           — merchant risk weight from Phase 1

GRAPH SPLIT — CRITICAL:
  Split chronologically by timestamp, NOT randomly:
  - Train: first 70% of transactions by time
  - Val:   next 15%
  - Test:  final 15%
  Random splitting causes data leakage (predicting past fraud
  using future data) — this will falsely inflate metrics.

Save outputs:
  data/processed/graph_train.pt
  data/processed/graph_val.pt
  data/processed/graph_test.pt

Print after build:
  - num_nodes, num_edges
  - fraud_rate (must be ~0.13%)
  - node_feature_dim, edge_feature_dim
  - graph density

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — CLASS BALANCING + MINI-BATCHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: src/data/smote.py + partitioner.py

GRAPH-SMOTE (src/data/smote.py):

The 0.13% fraud rate will cause the model to ignore fraud entirely
and hit 99.87% "accuracy" by predicting everything as legitimate.
Fix this at the data level with Graph-SMOTE:

Step 1 — Identify minority nodes
  - Find all nodes with fraud_label == 1
  - These are the seed nodes for synthesis

Step 2 — Synthetic node generation
  - For each seed fraud node, find its K=5 nearest neighbors
    in feature space (Euclidean distance on node features)
  - Generate synthetic fraud nodes by interpolating:
    x_synthetic = x_seed + λ × (x_neighbor − x_seed)
    where λ ~ Uniform(0, 1)

Step 3 — Edge generator
  - Train a small MLP (3 layers, 64 hidden units) to predict
    whether an edge should exist between two nodes
  - Input: concatenation of both node feature vectors
  - Output: sigmoid probability of edge existence
  - Train on real edges (positive) + random non-edges (negative)
  - Use this MLP to connect synthetic nodes back into the main graph

Step 4 — Validation
  - After SMOTE, target fraud rate: 1%–5%
  - Verify synthetic nodes have realistic feature distributions
  - Save augmented graph to data/processed/graph_train_smote.pt

METIS PARTITIONING (src/data/partitioner.py):

The full UPI graph cannot fit in GPU memory. Use METIS to partition
it into mini-batches that preserve graph structure:

  - Import pymetis (install: pip install pymetis)
  - Partition the training graph into N=50 subgraph clusters
  - Each cluster is a tightly-connected community
  - During training, load one cluster at a time into GPU memory
  - This reduces VRAM usage by ~60% vs full-graph loading
  - NEVER use random node sampling — it severs edges and destroys
    the topological context the model needs

  Save cluster assignments to:
  data/processed/cluster_assignments.pkl

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 4 — SENTINEL-GAT MODEL (CORE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: src/models/gat.py

This is the most critical component. Implement exactly as specified.

ARCHITECTURE:

class SentinelGAT(torch.nn.Module):
  - 2x Graph Attention layers (GATConv from PyG)
  - hidden_channels = 64
  - heads = 4
  - dropout = 0.3 (applied to both attention coefficients AND hidden layers)

ATTENTION MECHANISM — DO NOT SIMPLIFY:

Standard GATConv only uses node features x_i and x_j in the
attention score. Sentinel-UPI MUST include edge features h_ij:

  e_ij = LeakyReLU( a^T [ W·x_i ‖ W·x_j ‖ W_e·h_ij ] )

Where:
  - W  = learnable node feature transformation matrix
  - W_e = learnable edge feature transformation matrix
  - ‖  = concatenation
  - a  = learnable attention vector
  - LeakyReLU negative_slope = 0.2

Normalize with softmax over neighborhood:
  α_ij = exp(e_ij) / Σ_{k ∈ N(i)} exp(e_ik)

Multi-head aggregation (concatenate, not average):
  h'_i = ‖_{k=1}^{K} σ( Σ_{j ∈ N(i)} α_ij^(k) · W^(k) · x_j )

CLASSIFICATION HEAD:
  - Concatenate source embedding h'_u, dest embedding h'_v,
    and raw edge features e_uv
  - Pass through MLP (2 layers, 128 → 64 → 1)
  - Output: ŷ_uv = Sigmoid(MLP(h'_u ‖ h'_v ‖ e_uv))
  - This is the fraud probability for the transaction edge

WHY edge features in attention matters:
  A transaction to a new device (device_mismatch=1) should spike
  the attention weight even if the node itself looks normal.
  Without edge features, this signal is invisible to the model.
  This is what gives Sentinel-UPI its 4.2% F1 advantage over
  standard GraphSAGE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 5 — TRAINING LOOP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: src/models/focal_loss.py + src/training/train.py

FOCAL LOSS (src/models/focal_loss.py):

Standard CrossEntropy is useless at 0.13% fraud rate — the model
achieves 99.87% "accuracy" by predicting all-legitimate.
Focal Loss down-weights easy negatives and forces focus on hard fraud:

  FL(p_t) = −α_t × (1 − p_t)^γ × log(p_t)

Parameters (paper-specified, do not change):
  γ = 2.0   (focusing parameter — hard examples weighted more)
  α = 0.25  (class balancing weight for fraud class)

TRAINING LOOP (src/training/train.py):

Optimizer:
  AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)

Schedule:
  - Max 100 epochs
  - Early stopping: patience = 10 epochs on validation F1
  - Save best model checkpoint when val F1 improves

Per epoch:
  1. Iterate over METIS cluster batches (not full graph)
  2. Forward pass → fraud probability per edge
  3. Focal Loss on predicted vs true fraud labels
  4. Backward pass + AdamW step
  5. Log: train_loss, val_loss, val_F1, val_precision, val_recall

Log to console in this format every epoch:
  Epoch 042 | Loss: 0.0823 | Val F1: 0.941 | Prec: 0.968 | Rec: 0.916

FINAL EVALUATION (src/training/eval.py):

After training, evaluate on the held-out test set and report:
  - Precision, Recall, F1-score, AUC-ROC
  - Confusion matrix (TP, FP, TN, FN)
  - Inference latency (average ms per transaction)
  - Cost-sensitive loss:
      FP cost = ₹200 per incident (customer frustration + ops)
      FN cost = ₹15,000 per incident (fraud loss + regulatory fine)
      Report simulated monthly loss in ₹

ABLATION STUDIES (src/training/ablation.py):

Run these four experiments and report results as a markdown table:

  1. Drop edge features from attention → expect ~4.2% F1 drop
  2. CrossEntropy instead of Focal Loss → expect FP rate to spike
  3. Attention heads: test 1, 2, 4, 8 → confirm 4 is optimal
  4. GAT depth: test 1, 2, 3 layers → confirm 2 is optimal
     (3 layers will cause over-smoothing — node embeddings converge)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 6 — XAI EXPLAINABILITY LAYER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: src/models/explainer.py

Banks cannot legally block a transaction based on a black-box score.
RBI requires clear, auditable reasons for every blocked payment.

Use torch_geometric.explain.GNNExplainer on the trained model.

When risk_score >= 0.5, the explainer must output:

  {
    "txn_id": "TXN_001234",
    "risk_score": 0.94,
    "decision": "BLOCK",
    "top_features": [
      {"feature": "Transaction Velocity", "weight": 0.85},
      {"feature": "Time Delta (Δt)",      "weight": 0.72},
      {"feature": "In-Degree Centrality", "weight": 0.61},
      {"feature": "Log Amount",           "weight": 0.44},
      {"feature": "Account Age",          "weight": 0.21}
    ],
    "critical_edges": [
      {"from": "upi_A", "to": "upi_B", "weight": 0.91, "flag": "cycle"},
      {"from": "upi_B", "to": "upi_C", "weight": 0.87, "flag": "velocity"}
    ],
    "fraud_pattern": "Star topology — central node aggregating from 8 sources",
    "analyst_summary": "Account flagged due to 6 rapid micro-transactions
                        in 90 seconds routed through a known high-risk
                        merchant category. Matches triangular money mule
                        typology."
  }

This JSON accompanies every BLOCK decision from the inference API.
Fraud analysts see exactly which neighbor connections drove the flag.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 7 — REAL-TIME INFERENCE API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: src/inference/api.py + graph_cache.py + alerting.py

FASTAPI SERVER (api.py):

POST /score
  Input:
    {
      "txn_id": "TXN_001234",
      "src_upi": "user123@okicici",
      "dst_upi": "merchant456@oksbi",
      "amount": 2500.00,
      "timestamp": "2024-03-15T14:23:11Z",
      "merchant_type": "gaming_wallet",
      "device_id": "device_abc"
    }

  Pipeline (must complete in < 20ms total):
    1. Pull 2-hop temporal subgraph from Redis (last 24h only)
    2. Compute edge features for incoming transaction
    3. Run GAT forward pass
    4. Run GNN Explainer if risk_score >= 0.5
    5. If risk_score >= 0.5: trigger AWS SNS alert (async)
    6. Return response

  Output:
    {
      "txn_id": "TXN_001234",
      "risk_score": 0.94,
      "risk_level": "HIGH",        // HIGH ≥ 0.65, MEDIUM ≥ 0.35, LOW < 0.35
      "decision": "BLOCK",         // BLOCK if risk ≥ 0.5, ALLOW otherwise
      "latency_ms": 17.3,
      "explanation": { ...GNN Explainer output... }
    }

GET /health
  Returns: { "status": "ok", "model_version": "1.0.0", "uptime_s": 3600 }

GET /metrics
  Returns Prometheus-format metrics for Kubernetes health checks

GRAPH CACHE (graph_cache.py):

  - Connect to Redis Graph instance
  - On each /score request: pull the 2-hop neighborhood of src_upi
    filtered to the last 24 hours only (temporal subgraph sampling)
  - Cache node embeddings for repeat users with TTL = 300 seconds
  - This keeps cold-start requests under 20ms

ALERTING (alerting.py):

  - Publish to AWS SNS topic "sentinel-upi-fraud-alerts" when BLOCK
  - Message format: txn_id, risk_score, fraud_pattern, timestamp
  - Fire-and-forget (async) — do NOT await in the hot path
  - Simultaneously log to structured JSON log file

SECURITY REQUIREMENTS for the API:
  - All endpoints require API key header: X-API-Key
  - Rate limit: max 10,000 requests/minute per key
  - Input validation on all fields (Pydantic models)
  - No secrets or model weights in logs
  - HTTPS only in production

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 8 — CONTAINERIZATION & ORCHESTRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: docker/Dockerfile + docker-compose.yml + k8s/

DOCKERFILE:
  - Base: python:3.11-slim
  - Install PyTorch with CUDA 12.x support
  - Install PyTorch Geometric + all dependencies
  - Copy trained model weights to /app/models/
  - Expose port 8000
  - CMD: uvicorn src.inference.api:app --host 0.0.0.0 --port 8000

DOCKER-COMPOSE:
  Services:
    - sentinel-api: the inference server
    - redis-graph:  RedisGraph instance for subgraph cache
    - prometheus:   metrics scraping
    - grafana:      metrics dashboard

KUBERNETES (k8s/deployment.yaml):
  - Deployment: 3 replicas, resource limits 2CPU/4GB RAM per pod
  - GPU node selector for inference pods
  - Liveness probe: GET /health every 30s
  - Readiness probe: GET /health before routing traffic

KUBERNETES HPA (k8s/hpa.yaml):
  - Horizontal Pod Autoscaler
  - Min replicas: 3, Max replicas: 20
  - Scale up when CPU > 70% OR custom metric "inference_queue_depth" > 100
  - This handles UPI traffic spikes (holidays, sale events)

Stress test targets (verify with locust or k6):
  - Normal load (500 TPS): latency < 18ms, 0 errors
  - Peak load (5000 TPS): latency < 30ms, still under 100ms SLA

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 9 — ANALYTICS DASHBOARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: src/dashboard/

Build a React + Recharts dashboard with 4 tabs:

Tab 1 — Overview
  - KPI cards: Total Transactions, Fraud Flagged, Amount Blocked, FPR
  - Stacked area chart: safe vs flagged transactions over last 30 minutes
  - Donut chart: Low / Medium / High risk distribution
  - Bar chart: top 6 most flagged users

Tab 2 — Live Monitor
  - Real-time transaction table (auto-refreshes every 2 seconds)
  - Color-coded rows: red = HIGH, yellow = MEDIUM, white = LOW
  - Columns: Time, TXN ID, User, Amount, Merchant, Risk Score bar, Flags, Decision
  - Risk score shown as a mini progress bar + number

Tab 3 — Fraud Analytics
  - Bar chart: alerts by merchant category
  - Scatter plot: amount vs risk score (red = actual fraud, blue = legitimate)
  - Horizontal bar: anomaly rule trigger frequency
  - Graph visualization: D3 force-directed layout of flagged money-mule
    star topologies (central node + aggregating micro-transactions)

Tab 4 — Performance Metrics
  - 5 metric cards: Accuracy, Precision, Recall, F1, False Positive Rate
  - Visual confusion matrix (TP/FP/TN/FN with color coding)
  - Detection rule weight breakdown (horizontal bars)
  - Cost-sensitive P&L: estimated ₹ saved vs ₹ lost per month

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 10 — BASELINE BENCHMARK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: src/training/ablation.py (extend with baselines)

Train and evaluate all 5 models on the same test set.
Output a markdown table to results/benchmark.md:

| Model            | Type         | Precision | Recall | F1    | AUC-ROC | Latency (ms) |
|------------------|--------------|-----------|--------|-------|---------|--------------|
| Logistic Reg.    | Statistical  | 0.821     | 0.548  | 0.657 | 0.76    | ~1ms         |
| Random Forest    | Ensemble     | 0.942     | 0.763  | 0.843 | 0.88    | ~3ms         |
| XGBoost          | Boosting     | 0.965     | 0.831  | 0.893 | 0.92    | ~8ms         |
| GraphSAGE        | Graph (GNN)  | 0.951     | 0.884  | 0.916 | 0.94    | ~15ms        |
| Sentinel-UPI     | Graph (GAT)  | ≥0.972    | ≥0.956 | ≥0.964| ≥0.98  | <20ms        |

These are the paper's reported numbers — your implementation should
match or exceed them. If Sentinel-UPI F1 < 0.95, debug in this order:
  1. Check edge features are in the attention (most common mistake)
  2. Confirm Focal Loss is active (not CrossEntropy)
  3. Verify no data leakage in the chronological split
  4. Check Graph-SMOTE ran correctly (fraud rate 1–5% in training set)
  5. Try reducing learning rate to 0.001

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TESTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write pytest tests for every module. Minimum coverage:

tests/test_data.py:
  - Fraud rate is between 0.10% and 0.20% after loading
  - No transaction exceeds ₹100,000 after clipping
  - Chronological split: test timestamps > val timestamps > train timestamps
  - Edge feature dimensions match config (5 features)
  - DFS cycle flag correctly identifies A→B→C→A patterns

tests/test_model.py:
  - Model output shape: (num_edges, 1), values in [0, 1]
  - Focal Loss > 0 and finite for both fraud and legitimate samples
  - Attention weights sum to 1.0 per node neighborhood (softmax check)
  - GNN Explainer returns explanation dict with required keys

tests/test_inference.py:
  - /score returns 200 with valid JSON
  - Risk score in [0.0, 1.0]
  - BLOCK decision when risk_score >= 0.5
  - Inference latency < 20ms (time 100 consecutive requests)
  - API rejects requests without X-API-Key header

Run all tests: pytest tests/ -v --tb=short
All tests must pass before any phase is considered complete.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPLEMENTATION ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Execute phases strictly in this order. Do not skip ahead.
After each phase, run its tests before proceeding.

1. Project scaffold + CLAUDE.md + requirements.txt
2. paysim_loader.py → validate fraud rate + UPI adaptations
3. graph_builder.py + feature_engineer.py → validate graph stats
4. smote.py → validate post-SMOTE fraud rate
5. partitioner.py → validate 50 METIS clusters
6. focal_loss.py → unit test with synthetic data
7. gat.py → unit test forward pass shapes
8. train.py → run training, target val F1 > 0.95 by epoch 50
9. eval.py + ablation.py → benchmark vs baselines
10. explainer.py → validate explanation JSON structure
11. api.py + graph_cache.py + alerting.py → validate < 20ms latency
12. Dockerfile + docker-compose.yml → docker-compose up
13. k8s/ manifests → kubectl apply
14. Dashboard → all 4 tabs functional with live data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEGIN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Start with Phase 1. Create the project scaffold and
requirements.txt first, then implement paysim_loader.py.
Think through the UPI adaptations carefully before coding.
After each file, run its tests and confirm they pass.