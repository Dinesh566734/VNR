import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.dashboard.backend import create_dashboard_app


ROOT = Path(__file__).resolve().parents[1]


def _sample_snapshot() -> dict:
    return {
        "metadata": {
            "source": "unit-test",
            "generated_at": "2026-03-23T16:30:00+00:00",
            "warning": None
        },
        "overview": {
            "kpis": [{"label": "Total Transactions", "value": 8}],
            "timeline": [{"minute": "16:00", "safe": 4, "flagged": 1}],
            "risk_distribution": [{"name": "HIGH", "value": 2}],
            "top_flagged_users": [{"upi_id": "userA@okaxis", "count": 2}]
        },
        "live": {
            "updated_at": "2026-03-23T16:30:00+00:00",
            "transactions": [
                {
                    "time": "16:30:00",
                    "txn_id": "TXN_1001",
                    "user": "userA@okaxis",
                    "merchant": "mgame@ibl",
                    "merchant_type": "gaming_wallet",
                    "amount": 4200.0,
                    "risk_score": 0.91,
                    "risk_level": "HIGH",
                    "flags": ["velocity_burst"],
                    "decision": "BLOCK",
                    "actual_label": "FRAUD"
                }
            ]
        },
        "analytics": {
            "alerts_by_merchant": [{"merchant_type": "gaming_wallet", "count": 2}],
            "amount_vs_risk": [{"amount": 4200.0, "risk_score": 0.91, "label": "Fraud"}],
            "anomaly_rules": [{"rule": "Velocity Burst", "count": 3}],
            "network": {
                "nodes": [{"id": "userA@okaxis", "type": "user", "risk_level": "HIGH"}],
                "links": []
            }
        },
        "performance": {
            "cards": [{"label": "F1 Score", "value": 0.81}],
            "confusion_matrix": [{"label": "TP", "value": 41}],
            "rule_weights": [{"rule": "Velocity Burst", "weight": 0.4}],
            "pnl": {"saved_inr": 10000.0, "lost_inr": 1200.0, "net_inr": 8800.0},
            "training_summary": {"best_epoch": 12, "best_val_f1": 0.81, "adapted_rows": 265},
            "benchmark_models": [
                {
                    "name": "Logistic Reg.",
                    "type": "Statistical",
                    "precision": 0.0007,
                    "recall": 1.0,
                    "f1_score": 0.0014,
                    "auc_roc": 0.7380,
                    "latency_ms": 0.0002,
                },
                {
                    "name": "Random Forest",
                    "type": "Ensemble",
                    "precision": 0.0020,
                    "recall": 0.1111,
                    "f1_score": 0.0039,
                    "auc_roc": 0.7642,
                    "latency_ms": 0.0024,
                },
                {
                    "name": "XGBoost",
                    "type": "Boosting",
                    "precision": 0.0006,
                    "recall": 0.6296,
                    "f1_score": 0.0012,
                    "auc_roc": 0.3459,
                    "latency_ms": 0.0004,
                },
                {
                    "name": "GraphSAGE",
                    "type": "Graph (GNN)",
                    "precision": 0.0022,
                    "recall": 0.1111,
                    "f1_score": 0.0043,
                    "auc_roc": 0.4138,
                    "latency_ms": 0.0008,
                },
                {
                    "name": "Sentinel-UPI",
                    "type": "Graph (GAT)",
                    "precision": 0.0016,
                    "recall": 0.8148,
                    "f1_score": 0.0032,
                    "auc_roc": 0.7777,
                    "latency_ms": 0.0064,
                },
            ],
        }
    }


def test_dashboard_backend_exposes_all_phase_9_endpoints() -> None:
    client = TestClient(create_dashboard_app(snapshot_provider=_sample_snapshot))

    response = client.get("/dashboard/snapshot")

    assert response.status_code == 200
    payload = response.json()
    assert {"metadata", "overview", "live", "analytics", "performance"} <= set(payload)


def test_dashboard_backend_tab_endpoints_return_expected_shapes() -> None:
    client = TestClient(create_dashboard_app(snapshot_provider=_sample_snapshot))

    overview = client.get("/dashboard/overview").json()
    live = client.get("/dashboard/live").json()
    analytics = client.get("/dashboard/analytics").json()
    performance = client.get("/dashboard/performance").json()
    health = client.get("/dashboard/health").json()

    assert overview["kpis"][0]["label"] == "Total Transactions"
    assert live["transactions"][0]["decision"] == "BLOCK"
    assert analytics["network"]["nodes"][0]["id"] == "userA@okaxis"
    assert performance["cards"][0]["label"] == "F1 Score"
    assert len(performance["benchmark_models"]) == 5
    assert performance["benchmark_models"][-1]["name"] == "Sentinel-UPI"
    assert health["status"] == "ok"


def test_dashboard_frontend_declares_react_recharts_and_d3_stack() -> None:
    package_json = json.loads(
        (ROOT / "src" / "dashboard" / "frontend" / "package.json").read_text(encoding="utf-8")
    )

    assert package_json["dependencies"]["react"].startswith("^18")
    assert "recharts" in package_json["dependencies"]
    assert "d3-force" in package_json["dependencies"]
    assert package_json["scripts"]["build"] == "vite build"


def test_dashboard_frontend_includes_the_four_required_tabs() -> None:
    app_jsx = (
        ROOT / "src" / "dashboard" / "frontend" / "src" / "App.jsx"
    ).read_text(encoding="utf-8")

    assert "Overview" in app_jsx
    assert "Live Monitor" in app_jsx
    assert "Fraud Analytics" in app_jsx
    assert "Performance" in app_jsx
    assert "forceSimulation" in app_jsx
