from fastapi.testclient import TestClient

from src.inference.api import create_app


class _StubService:
    def score_transaction(self, payload: object) -> object:
        return type(
            "Result",
            (),
            {
                "txn_id": getattr(payload, "txn_id"),
                "risk_score": 0.2,
                "risk_level": "LOW",
                "decision": "ALLOW",
                "latency_ms": 1.0,
                "explanation": None,
            },
        )()


def test_health_endpoint_returns_status() -> None:
    client = TestClient(create_app(inference_service=_StubService(), api_keys={"test-key"}))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_version" in body
    assert "uptime_s" in body


def test_metrics_endpoint_returns_prometheus_payload() -> None:
    client = TestClient(create_app(inference_service=_StubService(), api_keys={"test-key"}))

    client.post(
        "/score",
        json={
            "txn_id": "TXN_METRICS_001",
            "src_upi": "user123@okicici",
            "dst_upi": "merchant456@oksbi",
            "amount": 2500.0,
            "timestamp": "2024-03-15T14:23:11Z",
            "merchant_type": "standard_retail",
            "device_id": "device_abc",
        },
        headers={"X-API-Key": "test-key"},
    )
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "sentinel_upi_score_requests_total" in response.text
