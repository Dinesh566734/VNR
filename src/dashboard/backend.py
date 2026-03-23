"""Analytics dashboard backend for Sentinel-UPI."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from torch_geometric.data import HeteroData

from src.data.paysim_loader import PROJECT_ROOT
from src.models.gat import SentinelGAT


DashboardSnapshotProvider = Callable[[], dict[str, Any]]
BLOCK_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 0.65
MEDIUM_RISK_THRESHOLD = 0.35
FP_COST_INR = 200.0
FN_COST_INR = 15000.0
MERCHANT_WEIGHT_MAPPING = {
    0.5: "utility_government",
    1.0: "standard_retail",
    1.5: "gaming_wallet",
    2.0: "crypto_offshore",
}


def create_dashboard_app(
    *,
    snapshot_provider: DashboardSnapshotProvider | None = None,
    processed_dir: str | Path | None = None,
) -> FastAPI:
    """Create the dashboard API application."""

    app = FastAPI(title="Sentinel-UPI Dashboard", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.snapshot_provider = snapshot_provider or (
        lambda: load_dashboard_snapshot(processed_dir=processed_dir)
    )

    @app.get("/dashboard/health")
    async def health() -> dict[str, Any]:
        snapshot = app.state.snapshot_provider()
        return {
            "status": "ok",
            "generated_at": snapshot["metadata"]["generated_at"],
            "source": snapshot["metadata"]["source"],
        }

    @app.get("/dashboard/snapshot")
    async def snapshot() -> dict[str, Any]:
        return app.state.snapshot_provider()

    @app.get("/dashboard/overview")
    async def overview() -> dict[str, Any]:
        return app.state.snapshot_provider()["overview"]

    @app.get("/dashboard/live")
    async def live() -> dict[str, Any]:
        return app.state.snapshot_provider()["live"]

    @app.get("/dashboard/analytics")
    async def analytics() -> dict[str, Any]:
        return app.state.snapshot_provider()["analytics"]

    @app.get("/dashboard/performance")
    async def performance() -> dict[str, Any]:
        return app.state.snapshot_provider()["performance"]

    return app


def load_dashboard_snapshot(
    *,
    processed_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load and cache dashboard data from processed artifacts, with safe fallback."""

    resolved_dir = _resolve_processed_dir(processed_dir)
    if resolved_dir is None:
        return _fallback_snapshot("No processed training run found.")

    try:
        return _load_dashboard_snapshot_cached(str(resolved_dir.resolve()))
    except Exception as exc:  # pragma: no cover - exercised only on artifact failures
        return _fallback_snapshot(f"Artifact loading failed: {exc}")


@lru_cache(maxsize=4)
def _load_dashboard_snapshot_cached(processed_dir: str) -> dict[str, Any]:
    run_dir = Path(processed_dir)
    evaluation = _read_json_if_exists(run_dir / "test_evaluation.json")
    training_summary = _read_json_if_exists(run_dir / "training_summary.json")
    graph = _load_graph_for_dashboard(run_dir)
    records = _build_transaction_records(graph=graph, checkpoint_path=run_dir / "sentinel_gat_best.pt")

    snapshot = _build_snapshot_from_records(
        records=records,
        evaluation=evaluation,
        training_summary=training_summary,
        metadata={
            "source": str(run_dir),
            "generated_at": datetime.now(UTC).isoformat(),
            "checkpoint_path": str((run_dir / "sentinel_gat_best.pt").resolve()),
            "graph_path": str((run_dir / "graph_test.pt").resolve()),
            "record_count": len(records),
        },
    )
    snapshot["metadata"]["warning"] = None
    return snapshot


def _resolve_processed_dir(processed_dir: str | Path | None) -> Path | None:
    if processed_dir is not None:
        resolved = Path(processed_dir)
        return resolved if resolved.exists() else None

    processed_root = PROJECT_ROOT / "data" / "processed"
    preferred = processed_root / "real_run_150k"
    if preferred.is_dir():
        return preferred

    candidate_dirs = [
        path
        for path in processed_root.glob("*")
        if path.is_dir() and (path / "training_summary.json").is_file()
    ]
    if candidate_dirs:
        return max(candidate_dirs, key=lambda path: path.stat().st_mtime)

    if (processed_root / "graph_test.pt").is_file():
        return processed_root
    return None


def _load_graph_for_dashboard(run_dir: Path) -> HeteroData:
    candidate_paths = [
        run_dir / "graph_test.pt",
        run_dir / "graph_val.pt",
        run_dir / "graph_train_smote.pt",
        run_dir / "graph_train.pt",
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return torch.load(candidate, weights_only=False)
    raise FileNotFoundError(f"No graph artifacts found in {run_dir}")


def _build_transaction_records(
    *,
    graph: HeteroData,
    checkpoint_path: Path | None,
) -> list[dict[str, Any]]:
    risk_scores = _predict_edge_scores(graph=graph, checkpoint_path=checkpoint_path)
    records: list[dict[str, Any]] = []

    for edge_type in graph.edge_types:
        edge_store = graph[edge_type]
        probabilities = risk_scores.get(edge_type)
        if edge_store.edge_index.numel() == 0:
            continue

        for edge_index in range(edge_store.edge_index.shape[1]):
            src_index = int(edge_store.edge_index[0, edge_index].item())
            dst_index = int(edge_store.edge_index[1, edge_index].item())
            timestamp = datetime.fromtimestamp(
                int(edge_store.timestamp[edge_index].item()),
                tz=UTC,
            )
            amount = float(torch.expm1(edge_store.edge_attr[edge_index, 0]).item())
            delta_t = float(edge_store.edge_attr[edge_index, 1].item())
            cycle_flag = bool(edge_store.edge_attr[edge_index, 3].item() >= 0.5)
            mcc_weight = float(edge_store.edge_attr[edge_index, 4].item())
            risk_score = (
                float(probabilities[edge_index].item())
                if probabilities is not None
                else _heuristic_risk_score(
                    amount=amount,
                    delta_t=delta_t,
                    cycle_flag=cycle_flag,
                    mcc_weight=mcc_weight,
                )
            )
            merchant_type = _merchant_type_for_edge(edge_type=edge_type, mcc_weight=mcc_weight)
            risk_level = _risk_level(risk_score)
            decision = "BLOCK" if risk_score >= BLOCK_THRESHOLD else "ALLOW"
            flags = _derive_flags(
                amount=amount,
                delta_t=delta_t,
                cycle_flag=cycle_flag,
                mcc_weight=mcc_weight,
                risk_score=risk_score,
            )

            records.append(
                {
                    "timestamp": timestamp,
                    "timestamp_label": timestamp.strftime("%H:%M:%S"),
                    "txn_id": edge_store.txn_id[edge_index],
                    "src_upi": graph[edge_type[0]].upi_id[src_index],
                    "dst_upi": graph[edge_type[2]].upi_id[dst_index],
                    "amount": round(amount, 2),
                    "merchant_type": merchant_type,
                    "risk_score": round(risk_score, 4),
                    "risk_level": risk_level,
                    "decision": decision,
                    "is_fraud": bool(edge_store.y[edge_index].item() >= 0.5),
                    "flags": flags,
                    "relation": edge_type[1],
                }
            )

    records.sort(key=lambda item: item["timestamp"])
    return records


def _predict_edge_scores(
    *,
    graph: HeteroData,
    checkpoint_path: Path | None,
) -> dict[tuple[str, str, str], torch.Tensor]:
    model = SentinelGAT.from_graph(graph)
    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        outputs = model(graph)
    return {
        edge_type: outputs[edge_type].reshape(-1).detach().cpu()
        for edge_type in graph.edge_types
    }


def _build_snapshot_from_records(
    *,
    records: list[dict[str, Any]],
    evaluation: dict[str, Any],
    training_summary: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    overview = _build_overview_section(records=records, evaluation=evaluation)
    live = _build_live_section(records=records)
    analytics = _build_analytics_section(records=records)
    performance = _build_performance_section(
        records=records,
        evaluation=evaluation,
        training_summary=training_summary,
    )
    return {
        "metadata": metadata,
        "overview": overview,
        "live": live,
        "analytics": analytics,
        "performance": performance,
    }


def _build_overview_section(
    *,
    records: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    total_transactions = len(records)
    blocked_transactions = [record for record in records if record["decision"] == "BLOCK"]
    amount_blocked = round(sum(record["amount"] for record in blocked_transactions), 2)
    risk_distribution = Counter(record["risk_level"] for record in records)
    flagged_user_counts = Counter(record["src_upi"] for record in blocked_transactions)

    return {
        "kpis": [
            {"label": "Total Transactions", "value": total_transactions},
            {"label": "Fraud Flagged", "value": len(blocked_transactions)},
            {"label": "Amount Blocked", "value": amount_blocked, "prefix": "INR "},
            {
                "label": "False Positive Rate",
                "value": round(float(evaluation.get("false_positive_rate", 0.0)) * 100.0, 2),
                "suffix": "%",
            },
        ],
        "timeline": _build_last_thirty_minutes(records),
        "risk_distribution": [
            {"name": "LOW", "value": risk_distribution.get("LOW", 0)},
            {"name": "MEDIUM", "value": risk_distribution.get("MEDIUM", 0)},
            {"name": "HIGH", "value": risk_distribution.get("HIGH", 0)},
        ],
        "top_flagged_users": [
            {"upi_id": upi_id, "count": count}
            for upi_id, count in flagged_user_counts.most_common(6)
        ],
    }


def _build_live_section(
    *,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    latest_records = list(reversed(records[-50:]))
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "transactions": [
            {
                "time": record["timestamp_label"],
                "txn_id": record["txn_id"],
                "user": record["src_upi"],
                "merchant": record["dst_upi"],
                "merchant_type": record["merchant_type"],
                "amount": record["amount"],
                "risk_score": record["risk_score"],
                "risk_level": record["risk_level"],
                "flags": record["flags"],
                "decision": record["decision"],
                "actual_label": "FRAUD" if record["is_fraud"] else "LEGIT",
            }
            for record in latest_records
        ],
    }


def _build_analytics_section(
    *,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    blocked_records = [record for record in records if record["decision"] == "BLOCK"]
    sampled_records = _sample_records_for_scatter(records, limit=240)
    rule_counts = Counter(flag for record in records for flag in record["flags"])
    merchant_counts = Counter(record["merchant_type"] for record in blocked_records)

    return {
        "alerts_by_merchant": [
            {"merchant_type": merchant_type, "count": count}
            for merchant_type, count in merchant_counts.most_common()
        ],
        "amount_vs_risk": [
            {
                "amount": record["amount"],
                "risk_score": record["risk_score"],
                "label": "Fraud" if record["is_fraud"] else "Legitimate",
                "merchant_type": record["merchant_type"],
            }
            for record in sampled_records
        ],
        "anomaly_rules": [
            {"rule": _humanize_flag(flag), "count": count}
            for flag, count in rule_counts.most_common(6)
        ],
        "network": _build_network_payload(blocked_records or _highest_risk_records(records, limit=12)),
    }


def _build_performance_section(
    *,
    records: list[dict[str, Any]],
    evaluation: dict[str, Any],
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    confusion = evaluation.get(
        "confusion_matrix",
        {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0},
    )
    tp = int(confusion.get("true_positive", 0))
    fp = int(confusion.get("false_positive", 0))
    tn = int(confusion.get("true_negative", 0))
    fn = int(confusion.get("false_negative", 0))
    total = max(1, tp + fp + tn + fn)
    accuracy = (tp + tn) / total
    rule_weights = _rule_weight_breakdown(records)
    saved_inr = tp * FN_COST_INR
    lost_inr = fn * FN_COST_INR + fp * FP_COST_INR

    return {
        "cards": [
            {"label": "Accuracy", "value": round(accuracy, 4)},
            {"label": "Precision", "value": round(float(evaluation.get("precision", 0.0)), 4)},
            {"label": "Recall", "value": round(float(evaluation.get("recall", 0.0)), 4)},
            {"label": "F1 Score", "value": round(float(evaluation.get("f1_score", 0.0)), 4)},
            {
                "label": "False Positive Rate",
                "value": round(float(evaluation.get("false_positive_rate", 0.0)), 4),
            },
        ],
        "confusion_matrix": [
            {"label": "TP", "value": tp},
            {"label": "FP", "value": fp},
            {"label": "TN", "value": tn},
            {"label": "FN", "value": fn},
        ],
        "rule_weights": rule_weights,
        "pnl": {
            "saved_inr": round(saved_inr, 2),
            "lost_inr": round(lost_inr, 2),
            "net_inr": round(saved_inr - lost_inr, 2),
            "reported_monthly_loss_inr": round(float(evaluation.get("monthly_loss_inr", 0.0)), 2),
        },
        "training_summary": {
            "best_epoch": training_summary.get("best_epoch", 0),
            "best_val_f1": training_summary.get("best_val_f1", 0.0),
            "adapted_rows": training_summary.get("adapted_rows", 0),
        },
    }


def _build_last_thirty_minutes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []

    latest_timestamp = records[-1]["timestamp"]
    window_start = latest_timestamp - timedelta(minutes=29)
    buckets = {
        (window_start + timedelta(minutes=offset)).strftime("%H:%M"): {"safe": 0, "flagged": 0}
        for offset in range(30)
    }

    for record in records:
        if record["timestamp"] < window_start:
            continue
        key = record["timestamp"].strftime("%H:%M")
        if key not in buckets:
            continue
        bucket = buckets[key]
        if record["decision"] == "BLOCK":
            bucket["flagged"] += 1
        else:
            bucket["safe"] += 1

    return [
        {"minute": minute, "safe": counts["safe"], "flagged": counts["flagged"]}
        for minute, counts in buckets.items()
    ]


def _sample_records_for_scatter(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return records
    step = max(1, len(records) // limit)
    sampled = records[::step][:limit]
    highlighted = _highest_risk_records(records, limit=min(30, limit // 4))
    deduped = {record["txn_id"]: record for record in [*sampled, *highlighted]}
    return list(deduped.values())[:limit]


def _highest_risk_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: record["risk_score"], reverse=True)[:limit]


def _build_network_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"nodes": [], "links": []}

    connectivity = Counter()
    for record in records:
        connectivity[record["src_upi"]] += 1
        connectivity[record["dst_upi"]] += 1
    central_node = connectivity.most_common(1)[0][0]

    focused_records = [
        record
        for record in records
        if record["src_upi"] == central_node or record["dst_upi"] == central_node
    ] or records[:12]

    nodes: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    for record in focused_records[:12]:
        nodes.setdefault(
            record["src_upi"],
            {"id": record["src_upi"], "type": "user", "risk_level": record["risk_level"]},
        )
        nodes.setdefault(
            record["dst_upi"],
            {
                "id": record["dst_upi"],
                "type": "user" if record["merchant_type"] == "peer_to_peer" else "merchant",
                "risk_level": record["risk_level"],
            },
        )
        links.append(
            {
                "source": record["src_upi"],
                "target": record["dst_upi"],
                "risk_score": record["risk_score"],
                "decision": record["decision"],
            }
        )
    return {"nodes": list(nodes.values()), "links": links}


def _rule_weight_breakdown(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocked_records = [record for record in records if record["decision"] == "BLOCK"]
    active_records = blocked_records or records
    counts = Counter(flag for record in active_records for flag in record["flags"])
    total = max(1, sum(counts.values()))
    return [
        {"rule": _humanize_flag(flag), "weight": round(count / total, 4)}
        for flag, count in counts.most_common(6)
    ]


def _merchant_type_for_edge(
    *,
    edge_type: tuple[str, str, str],
    mcc_weight: float,
) -> str:
    if edge_type[2] == "user":
        return "peer_to_peer"
    rounded_weight = round(mcc_weight, 1)
    return MERCHANT_WEIGHT_MAPPING.get(rounded_weight, "standard_retail")


def _risk_level(risk_score: float) -> str:
    if risk_score >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    if risk_score >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _derive_flags(
    *,
    amount: float,
    delta_t: float,
    cycle_flag: bool,
    mcc_weight: float,
    risk_score: float,
) -> list[str]:
    flags: list[str] = []
    if delta_t <= 120.0:
        flags.append("velocity_burst")
    if cycle_flag:
        flags.append("cycle_pattern")
    if mcc_weight >= 1.5:
        flags.append("high_risk_merchant")
    if amount >= 75000.0:
        flags.append("large_amount")
    if risk_score >= HIGH_RISK_THRESHOLD:
        flags.append("attention_spike")
    return flags or ["baseline_pattern"]


def _heuristic_risk_score(
    *,
    amount: float,
    delta_t: float,
    cycle_flag: bool,
    mcc_weight: float,
) -> float:
    score = 0.10
    score += min(0.25, amount / 100000.0 * 0.25)
    if delta_t <= 120.0:
        score += 0.20
    if cycle_flag:
        score += 0.25
    if mcc_weight >= 1.5:
        score += 0.20
    return float(min(score, 0.99))


def _humanize_flag(flag: str) -> str:
    return flag.replace("_", " ").title()


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fallback_snapshot(reason: str) -> dict[str, Any]:
    sample_records = _sample_dashboard_records()
    snapshot = _build_snapshot_from_records(
        records=sample_records,
        evaluation={
            "precision": 0.84,
            "recall": 0.79,
            "f1_score": 0.81,
            "false_positive_rate": 0.07,
            "monthly_loss_inr": 184000.0,
            "confusion_matrix": {
                "true_positive": 41,
                "false_positive": 9,
                "true_negative": 204,
                "false_negative": 11,
            },
        },
        training_summary={
            "best_epoch": 12,
            "best_val_f1": 0.81,
            "adapted_rows": 265,
        },
        metadata={
            "source": "fallback",
            "generated_at": datetime.now(UTC).isoformat(),
            "checkpoint_path": None,
            "graph_path": None,
            "record_count": len(sample_records),
            "warning": reason,
        },
    )
    snapshot["metadata"]["warning"] = reason
    return snapshot


def _sample_dashboard_records() -> list[dict[str, Any]]:
    base_time = datetime(2024, 3, 15, 14, 0, tzinfo=UTC)
    raw_rows = [
        ("TXN_1001", "userA@okaxis", "mgame@ibl", 4200.0, "gaming_wallet", 0.91, True, ["velocity_burst", "high_risk_merchant"]),
        ("TXN_1002", "userB@ybl", "merchantX@oksbi", 1800.0, "standard_retail", 0.28, False, ["baseline_pattern"]),
        ("TXN_1003", "userC@okicici", "userD@oksbi", 95000.0, "peer_to_peer", 0.77, True, ["large_amount", "attention_spike"]),
        ("TXN_1004", "userA@okaxis", "utility@okhdfcbank", 600.0, "utility_government", 0.22, False, ["baseline_pattern"]),
        ("TXN_1005", "userE@okhdfcbank", "mcrypto@ibl", 12000.0, "crypto_offshore", 0.88, True, ["cycle_pattern", "high_risk_merchant"]),
        ("TXN_1006", "userF@oksbi", "merchantY@paytm", 2200.0, "standard_retail", 0.34, False, ["baseline_pattern"]),
        ("TXN_1007", "userG@okicici", "mgame@ibl", 800.0, "gaming_wallet", 0.65, True, ["velocity_burst", "attention_spike"]),
        ("TXN_1008", "userH@ybl", "userI@okaxis", 76000.0, "peer_to_peer", 0.54, False, ["large_amount"]),
    ]
    records: list[dict[str, Any]] = []
    for offset, row in enumerate(raw_rows):
        txn_id, src_upi, dst_upi, amount, merchant_type, risk_score, is_fraud, flags = row
        timestamp = base_time + timedelta(minutes=offset * 4)
        records.append(
            {
                "timestamp": timestamp,
                "timestamp_label": timestamp.strftime("%H:%M:%S"),
                "txn_id": txn_id,
                "src_upi": src_upi,
                "dst_upi": dst_upi,
                "amount": amount,
                "merchant_type": merchant_type,
                "risk_score": risk_score,
                "risk_level": _risk_level(risk_score),
                "decision": "BLOCK" if risk_score >= BLOCK_THRESHOLD else "ALLOW",
                "is_fraud": is_fraud,
                "flags": flags,
                "relation": "synthetic",
            }
        )
    return records


app = create_dashboard_app()
