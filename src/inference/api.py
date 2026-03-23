"""FastAPI inference server for Sentinel-UPI."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from torch_geometric.data import HeteroData

from src.data.paysim_loader import PROJECT_ROOT, load_config
from src.models.explainer import SentinelExplainer
from src.models.gat import SentinelGAT
from .alerting import AlertingService
from .graph_cache import GraphCache


MerchantType = Literal[
    "peer_to_peer",
    "utility_government",
    "standard_retail",
    "gaming_wallet",
    "crypto_offshore",
]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
Decision = Literal["ALLOW", "BLOCK"]


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    txn_id: str = Field(min_length=3, max_length=128)
    src_upi: str = Field(min_length=3, max_length=128)
    dst_upi: str = Field(min_length=3, max_length=128)
    amount: float = Field(gt=0.0, le=1_000_000.0)
    timestamp: datetime
    merchant_type: MerchantType
    device_id: str = Field(min_length=3, max_length=128)


class ScoreResponse(BaseModel):
    txn_id: str
    risk_score: float
    risk_level: RiskLevel
    decision: Decision
    latency_ms: float
    explanation: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_s: int


@dataclass
class ScoreResult:
    txn_id: str
    risk_score: float
    risk_level: RiskLevel
    decision: Decision
    latency_ms: float
    explanation: dict[str, Any] | None


class RateLimiter:
    """Simple in-memory per-key rate limiter."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = limit_per_minute
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, api_key: str) -> bool:
        now = time.time()
        queue = self.events[api_key]
        while queue and queue[0] <= now - 60.0:
            queue.popleft()
        if len(queue) >= self.limit_per_minute:
            return False
        queue.append(now)
        return True


class InferenceService:
    """Score incoming transactions using the model, cache, and explainer."""

    def __init__(
        self,
        *,
        model: SentinelGAT,
        graph_cache: GraphCache,
        explainer: SentinelExplainer,
        alerting_service: AlertingService,
        config_path: str | None = None,
    ) -> None:
        self.config = load_config(config_path)
        self.model = model
        self.graph_cache = graph_cache
        self.explainer = explainer
        self.alerting_service = alerting_service
        self.model.eval()

    def score_transaction(self, request: ScoreRequest) -> ScoreResult:
        started_at = time.perf_counter()
        subgraph = self.graph_cache.get_temporal_subgraph(
            src_upi=request.src_upi,
            timestamp=request.timestamp,
            lookback_hours=24,
            hops=2,
        )
        working_graph, relation, edge_position, src_index, dst_index = self._graph_for_request(
            subgraph=subgraph,
            request=request,
        )

        with torch.no_grad():
            outputs = self.model(working_graph)
            risk_score = float(outputs[relation][edge_position].item())
            embeddings, _ = self.model.encode(working_graph)
            self.graph_cache.set_cached_embedding(request.src_upi, embeddings["user"][src_index])
            if relation[2] == "user":
                self.graph_cache.set_cached_embedding(request.dst_upi, embeddings["user"][dst_index])
            else:
                self.graph_cache.set_cached_embedding(request.dst_upi, embeddings["merchant"][dst_index])

        risk_level = self._risk_level(risk_score)
        decision = "BLOCK" if risk_score >= 0.5 else "ALLOW"
        explanation = None
        if decision == "BLOCK":
            explanation = self.explainer.explain_transaction(
                working_graph,
                edge_type=relation,
                edge_index=edge_position,
            )
            alert = AlertingService.build_alert(
                txn_id=request.txn_id,
                risk_score=risk_score,
                fraud_pattern=explanation["fraud_pattern"],
                timestamp=request.timestamp,
            )
            self.alerting_service.dispatch_background(alert)

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return ScoreResult(
            txn_id=request.txn_id,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            decision=decision,
            latency_ms=round(latency_ms, 3),
            explanation=explanation,
        )

    def _graph_for_request(
        self,
        *,
        subgraph: HeteroData,
        request: ScoreRequest,
    ) -> tuple[HeteroData, tuple[str, str, str], int, int, int]:
        graph = _clone_subgraph(subgraph)
        relation_names = self.config["graph"]["relations"]
        relation = (
            ("user", relation_names["p2p"], "user")
            if request.merchant_type == "peer_to_peer"
            else ("user", relation_names["merchant"], "merchant")
        )

        src_index = _ensure_node(
            graph=graph,
            node_type="user",
            upi_id=request.src_upi,
            feature_vector=self._default_node_features("user"),
        )
        destination_type = relation[2]
        dst_index = _ensure_node(
            graph=graph,
            node_type=destination_type,
            upi_id=request.dst_upi,
            feature_vector=self._default_node_features(destination_type),
        )

        request_ts = int(pd_timestamp(request.timestamp))
        delta_t = self._previous_transaction_delta(graph, src_index=src_index, request_ts=request_ts)
        temporal_decay = math_exp_decay(
            delta_t=delta_t,
            rate=float(self.config["features"]["edge"]["temporal_decay_rate"]),
        )
        cycle_flag = float(
            _would_close_user_cycle(graph, src_index=src_index, dst_index=dst_index)
            if destination_type == "user"
            else 0.0
        )
        mcc_weight = float(self.config["data"]["merchant_risk_weights"][request.merchant_type])
        edge_attr = torch.tensor(
            [
                math_log1p(request.amount),
                float(delta_t),
                temporal_decay,
                cycle_flag,
                mcc_weight,
            ],
            dtype=torch.float32,
        )

        edge_position = graph[relation].edge_index.shape[1]
        _append_edge(
            graph=graph,
            relation=relation,
            src_index=src_index,
            dst_index=dst_index,
            edge_attr=edge_attr,
            timestamp=request_ts,
            txn_id=request.txn_id,
        )
        return graph, relation, edge_position, src_index, dst_index

    def _default_node_features(self, node_type: str) -> torch.Tensor:
        source_graph = self.graph_cache.graph
        if source_graph is not None and source_graph[node_type].x.numel():
            return source_graph[node_type].x.mean(dim=0).detach().clone().float()
        fallback_dim = 5
        return torch.full((fallback_dim,), 0.5, dtype=torch.float32)

    def _previous_transaction_delta(
        self,
        graph: HeteroData,
        *,
        src_index: int,
        request_ts: int,
    ) -> float:
        latest_timestamp: int | None = None
        for relation in graph.edge_types:
            edge_store = graph[relation]
            if edge_store.edge_index.numel() == 0:
                continue
            src_mask = edge_store.edge_index[0] == src_index
            if src_mask.any():
                candidate = int(edge_store.timestamp[src_mask].max().item())
                if latest_timestamp is None or candidate > latest_timestamp:
                    latest_timestamp = candidate
        if latest_timestamp is None:
            return 86400.0
        return float(max(0, request_ts - latest_timestamp))

    @staticmethod
    def _risk_level(risk_score: float) -> RiskLevel:
        if risk_score >= 0.65:
            return "HIGH"
        if risk_score >= 0.35:
            return "MEDIUM"
        return "LOW"


def create_app(
    *,
    inference_service: InferenceService | None = None,
    api_keys: set[str] | None = None,
    rate_limit_per_minute: int = 10_000,
    environment: str = "development",
    model_version: str | None = None,
) -> FastAPI:
    """Create the FastAPI application with dependency injection for tests."""

    config = load_config()
    registry = CollectorRegistry()
    request_counter = Counter(
        "sentinel_upi_score_requests_total",
        "Total score requests",
        registry=registry,
    )
    blocked_counter = Counter(
        "sentinel_upi_blocked_transactions_total",
        "Total blocked transactions",
        registry=registry,
    )
    latency_histogram = Histogram(
        "sentinel_upi_score_latency_ms",
        "Score request latency in milliseconds",
        registry=registry,
    )

    app = FastAPI(title="Sentinel-UPI API", version=model_version or config["project"]["version"])
    app.state.started_at = time.time()
    app.state.environment = environment
    app.state.api_keys = api_keys or {"dev-secret-key"}
    app.state.rate_limiter = RateLimiter(rate_limit_per_minute)
    app.state.registry = registry
    app.state.request_counter = request_counter
    app.state.blocked_counter = blocked_counter
    app.state.latency_histogram = latency_histogram
    app.state.inference_service = inference_service or build_default_inference_service()
    app.state.model_version = model_version or config["project"]["version"]

    async def require_api_key(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> str:
        if app.state.environment.lower() == "production":
            proto = request.headers.get("x-forwarded-proto", request.url.scheme)
            if proto != "https":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="HTTPS is required in production.",
                )

        if not x_api_key or x_api_key not in app.state.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key.",
            )
        if not app.state.rate_limiter.allow(x_api_key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded.",
            )
        return x_api_key

    @app.post("/score", response_model=ScoreResponse)
    async def score_transaction(
        payload: ScoreRequest,
        _: str = Depends(require_api_key),
    ) -> ScoreResponse:
        service: InferenceService | None = app.state.inference_service
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference service is not configured.",
            )
        result = service.score_transaction(payload)
        app.state.request_counter.inc()
        app.state.latency_histogram.observe(result.latency_ms)
        if result.decision == "BLOCK":
            app.state.blocked_counter.inc()
        return ScoreResponse(
            txn_id=result.txn_id,
            risk_score=result.risk_score,
            risk_level=result.risk_level,
            decision=result.decision,
            latency_ms=result.latency_ms,
            explanation=result.explanation,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_version=app.state.model_version,
            uptime_s=int(time.time() - app.state.started_at),
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            content=generate_latest(app.state.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


def build_default_inference_service() -> InferenceService | None:
    """Best-effort default service from locally available processed graphs."""

    candidate_paths = [
        PROJECT_ROOT / "data" / "processed" / "graph_train_smote.pt",
        PROJECT_ROOT / "data" / "processed" / "graph_train.pt",
        PROJECT_ROOT / "data" / "processed" / "graph_val.pt",
        PROJECT_ROOT / "data" / "processed" / "graph_test.pt",
    ]
    graph = None
    for candidate in candidate_paths:
        if candidate.is_file():
            graph = torch.load(candidate, weights_only=False)
            break

    if graph is None:
        return None

    model = SentinelGAT.from_graph(graph)
    graph_cache = GraphCache(graph=graph)
    explainer = SentinelExplainer(model)
    alerting = AlertingService(enabled=False)
    return InferenceService(
        model=model,
        graph_cache=graph_cache,
        explainer=explainer,
        alerting_service=alerting,
    )


def pd_timestamp(timestamp: datetime) -> int:
    return int(timestamp.timestamp())


def math_log1p(amount: float) -> float:
    return float(torch.log1p(torch.tensor(amount, dtype=torch.float32)).item())


def math_exp_decay(*, delta_t: float, rate: float) -> float:
    return float(torch.exp(torch.tensor(-rate * delta_t, dtype=torch.float32)).item())


def _clone_subgraph(graph: HeteroData) -> HeteroData:
    clone = HeteroData()
    for node_type in ("user", "merchant"):
        clone[node_type].x = graph[node_type].x.clone()
        clone[node_type].num_nodes = graph[node_type].num_nodes
        clone[node_type].upi_id = list(getattr(graph[node_type], "upi_id", []))
        if hasattr(graph[node_type], "synthetic_mask"):
            clone[node_type].synthetic_mask = graph[node_type].synthetic_mask.clone()

    for edge_type in graph.edge_types:
        clone[edge_type].edge_index = graph[edge_type].edge_index.clone()
        clone[edge_type].edge_attr = graph[edge_type].edge_attr.clone()
        clone[edge_type].y = graph[edge_type].y.clone()
        clone[edge_type].timestamp = graph[edge_type].timestamp.clone()
        clone[edge_type].txn_id = list(getattr(graph[edge_type], "txn_id", []))
    return clone


def _ensure_node(
    *,
    graph: HeteroData,
    node_type: str,
    upi_id: str,
    feature_vector: torch.Tensor,
) -> int:
    existing_ids = list(graph[node_type].upi_id)
    if upi_id in existing_ids:
        return existing_ids.index(upi_id)

    graph[node_type].x = torch.cat([graph[node_type].x, feature_vector.reshape(1, -1)], dim=0)
    graph[node_type].num_nodes = graph[node_type].x.shape[0]
    graph[node_type].upi_id = existing_ids + [upi_id]
    if hasattr(graph[node_type], "synthetic_mask"):
        graph[node_type].synthetic_mask = torch.cat(
            [graph[node_type].synthetic_mask, torch.tensor([False], dtype=torch.bool)]
        )
    return graph[node_type].num_nodes - 1


def _append_edge(
    *,
    graph: HeteroData,
    relation: tuple[str, str, str],
    src_index: int,
    dst_index: int,
    edge_attr: torch.Tensor,
    timestamp: int,
    txn_id: str,
) -> None:
    edge_store = graph[relation]
    if not hasattr(edge_store, "edge_index"):
        edge_store.edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_store.edge_attr = torch.empty((0, edge_attr.shape[0]), dtype=torch.float32)
        edge_store.y = torch.empty((0,), dtype=torch.float32)
        edge_store.timestamp = torch.empty((0,), dtype=torch.long)
        edge_store.txn_id = []

    edge_store.edge_index = torch.cat(
        [edge_store.edge_index, torch.tensor([[src_index], [dst_index]], dtype=torch.long)],
        dim=1,
    )
    edge_store.edge_attr = torch.cat([edge_store.edge_attr, edge_attr.reshape(1, -1)], dim=0)
    edge_store.y = torch.cat([edge_store.y, torch.tensor([0.0], dtype=torch.float32)], dim=0)
    edge_store.timestamp = torch.cat(
        [edge_store.timestamp, torch.tensor([timestamp], dtype=torch.long)],
        dim=0,
    )
    edge_store.txn_id = list(edge_store.txn_id) + [txn_id]


def _would_close_user_cycle(graph: HeteroData, *, src_index: int, dst_index: int) -> bool:
    user_relations = [edge_type for edge_type in graph.edge_types if edge_type[2] == "user"]
    adjacency: dict[int, set[int]] = {}
    for relation in user_relations:
        edge_store = graph[relation]
        for source, destination in edge_store.edge_index.t().tolist():
            adjacency.setdefault(int(source), set()).add(int(destination))

    stack = [dst_index]
    seen = {dst_index}
    while stack:
        node = stack.pop()
        for neighbor in adjacency.get(node, set()):
            if neighbor == src_index:
                return True
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return False
