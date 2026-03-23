import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
import torch

from src.data.graph_builder import build_hetero_graph
from src.inference.alerting import AlertingService
from src.inference.api import InferenceService, ScoreRequest, create_app
from src.inference.graph_cache import GraphCache
from src.models.explainer import SentinelExplainer
from src.models.gat import SentinelGAT


def _sample_graph() -> object:
    transactions = pd.DataFrame(
        [
            {
                "txn_id": "TXN_001",
                "src_upi": "ua@okaxis",
                "dst_upi": "ub@oksbi",
                "amount_clipped": 1200.0,
                "timestamp": "2024-01-01T09:00:00",
                "merchant_type": "peer_to_peer",
                "mcc_weight": 1.0,
                "is_fraud": 0,
            },
            {
                "txn_id": "TXN_002",
                "src_upi": "ub@oksbi",
                "dst_upi": "mshop@paytm",
                "amount_clipped": 3100.0,
                "timestamp": "2024-01-01T09:10:00",
                "merchant_type": "standard_retail",
                "mcc_weight": 1.0,
                "is_fraud": 1,
            },
            {
                "txn_id": "TXN_003",
                "src_upi": "uc@ybl",
                "dst_upi": "mgame@ibl",
                "amount_clipped": 950.0,
                "timestamp": "2024-01-01T09:20:00",
                "merchant_type": "gaming_wallet",
                "mcc_weight": 1.5,
                "is_fraud": 1,
            },
        ]
    )
    return build_hetero_graph(transactions)


def test_graph_cache_returns_temporal_subgraph() -> None:
    graph = _sample_graph()
    cache = GraphCache(graph=graph)

    subgraph = cache.get_temporal_subgraph(
        src_upi="ua@okaxis",
        timestamp=datetime(2024, 1, 1, 10, 0, 0),
    )

    assert subgraph["user"].num_nodes >= 1
    assert sum(subgraph[edge_type].edge_index.shape[1] for edge_type in subgraph.edge_types) >= 1


def test_alerting_service_writes_structured_log(tmp_path: Path) -> None:
    log_path = tmp_path / "alerts.jsonl"
    service = AlertingService(enabled=True, log_path=log_path)
    alert = service.build_alert(
        txn_id="TXN_999",
        risk_score=0.91,
        fraud_pattern="velocity burst",
    )
    service.publish(alert)

    assert log_path.is_file()
    assert '"txn_id": "TXN_999"' in log_path.read_text(encoding="utf-8")


def test_inference_service_scores_transaction_in_range(tmp_path: Path) -> None:
    graph = _sample_graph()
    model = SentinelGAT.from_graph(graph)
    cache = GraphCache(graph=graph)
    explainer = SentinelExplainer(model, gnn_explainer_epochs=2)
    alerting = AlertingService(enabled=True, log_path=tmp_path / "alerts.jsonl")
    service = InferenceService(
        model=model,
        graph_cache=cache,
        explainer=explainer,
        alerting_service=alerting,
    )

    result = service.score_transaction(
        ScoreRequest(
            txn_id="TXN_SCORE_001",
            src_upi="ua@okaxis",
            dst_upi="mgame@ibl",
            amount=2500.0,
            timestamp=datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc),
            merchant_type="gaming_wallet",
            device_id="device_abc",
        )
    )

    assert 0.0 <= result.risk_score <= 1.0
    assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}
    assert result.decision in {"ALLOW", "BLOCK"}
    assert result.latency_ms >= 0.0


def test_score_endpoint_latency_and_auth() -> None:
    class StubService:
        def score_transaction(self, payload: object) -> object:
            return type(
                "Result",
                (),
                {
                    "txn_id": getattr(payload, "txn_id"),
                    "risk_score": 0.9,
                    "risk_level": "HIGH",
                    "decision": "BLOCK",
                    "latency_ms": 1.2,
                    "explanation": {"fraud_pattern": "stub"},
                },
            )()

    client = TestClient(
        create_app(
            inference_service=StubService(),
            api_keys={"test-key"},
            rate_limit_per_minute=10_000,
        )
    )
    payload = {
        "txn_id": "TXN_API_001",
        "src_upi": "user123@okicici",
        "dst_upi": "merchant456@oksbi",
        "amount": 2500.0,
        "timestamp": "2024-03-15T14:23:11Z",
        "merchant_type": "gaming_wallet",
        "device_id": "device_abc",
    }

    response = client.post("/score", json=payload)
    assert response.status_code == 401

    start = time.perf_counter()
    results = [
        client.post("/score", json=payload, headers={"X-API-Key": "test-key"})
        for _ in range(100)
    ]
    elapsed_ms_per_request = ((time.perf_counter() - start) * 1000.0) / 100.0

    assert all(result.status_code == 200 for result in results)
    assert all(0.0 <= result.json()["risk_score"] <= 1.0 for result in results)
    assert all(result.json()["decision"] == "BLOCK" for result in results)
    assert elapsed_ms_per_request < 20.0
