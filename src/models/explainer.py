"""Explainability layer for Sentinel-UPI transaction decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from src.data.feature_engineer import EDGE_FEATURE_COLUMNS, NODE_FEATURE_COLUMNS
from .gat import SentinelGAT


NODE_FEATURE_LABELS = {
    "account_age": "Account Age",
    "kyc_tier": "KYC Tier",
    "daily_avg_spend": "Daily Average Spend",
    "pagerank_local": "Localized PageRank",
    "betweenness_central": "Betweenness Centrality",
}
EDGE_FEATURE_LABELS = {
    "log_amount": "Log Amount",
    "delta_t": "Time Delta (Δt)",
    "temporal_decay": "Transaction Velocity",
    "dfs_cycle_flag": "Cycle Flag",
    "mcc_weight": "Merchant Risk Weight",
}


@dataclass
class FeatureContribution:
    feature: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "weight": round(self.weight, 4)}


@dataclass
class CriticalEdge:
    source: str
    destination: str
    weight: float
    flag: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.source,
            "to": self.destination,
            "weight": round(self.weight, 4),
            "flag": self.flag,
        }


class RelationPredictionWrapper(nn.Module):
    """Wrap SentinelGAT so PyG Explainer can target one relation at a time."""

    def __init__(self, model: SentinelGAT, relation: tuple[str, str, str]) -> None:
        super().__init__()
        self.model = model
        self.relation = relation

    def forward(
        self,
        x: dict[str, Tensor],
        edge_index: dict[tuple[str, str, str], Tensor],
        *,
        edge_attr: dict[tuple[str, str, str], Tensor],
    ) -> Tensor:
        graph = HeteroData()
        for node_type, features in x.items():
            graph[node_type].x = features
            graph[node_type].num_nodes = features.shape[0]
        for edge_type, relation_index in edge_index.items():
            graph[edge_type].edge_index = relation_index
            graph[edge_type].edge_attr = edge_attr[edge_type]

        outputs = self.model(graph)
        return outputs[self.relation].reshape(-1)


class SentinelExplainer:
    """Generate analyst-facing explanations for edge fraud decisions."""

    def __init__(
        self,
        model: SentinelGAT,
        *,
        decision_threshold: float = 0.5,
        gnn_explainer_epochs: int = 30,
        gnn_explainer_lr: float = 0.01,
    ) -> None:
        self.model = model
        self.decision_threshold = decision_threshold
        self.gnn_explainer_epochs = gnn_explainer_epochs
        self.gnn_explainer_lr = gnn_explainer_lr

    def explain_transaction(
        self,
        graph: HeteroData,
        *,
        edge_type: tuple[str, str, str],
        edge_index: int,
    ) -> dict[str, Any]:
        """Explain a specific transaction edge as a JSON-compatible dict."""

        risk_score = self._predict_edge_probability(graph, edge_type=edge_type, edge_index=edge_index)
        decision = "BLOCK" if risk_score >= self.decision_threshold else "ALLOW"
        explanation = self._run_gnn_explainer(graph, edge_type=edge_type, edge_index=edge_index)
        gradients = self._gradient_feature_importance(graph, edge_type=edge_type, edge_index=edge_index)

        top_features = self._top_features(gradients)
        critical_edges = self._critical_edges(
            graph=graph,
            edge_type=edge_type,
            edge_index=edge_index,
            explanation=explanation,
        )
        fraud_pattern = self._fraud_pattern(
            graph=graph,
            edge_type=edge_type,
            edge_index=edge_index,
            critical_edges=critical_edges,
        )

        return {
            "txn_id": graph[edge_type].txn_id[edge_index],
            "risk_score": round(risk_score, 4),
            "decision": decision,
            "top_features": [item.to_dict() for item in top_features],
            "critical_edges": [item.to_dict() for item in critical_edges],
            "fraud_pattern": fraud_pattern,
            "analyst_summary": self._analyst_summary(
                graph=graph,
                edge_type=edge_type,
                edge_index=edge_index,
                risk_score=risk_score,
                top_features=top_features,
                critical_edges=critical_edges,
                fraud_pattern=fraud_pattern,
            ),
        }

    def _predict_edge_probability(
        self,
        graph: HeteroData,
        *,
        edge_type: tuple[str, str, str],
        edge_index: int,
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(graph)
            return float(outputs[edge_type][edge_index].item())

    def _run_gnn_explainer(
        self,
        graph: HeteroData,
        *,
        edge_type: tuple[str, str, str],
        edge_index: int,
    ) -> Any | None:
        wrapper = RelationPredictionWrapper(self.model, relation=edge_type)
        wrapper.eval()
        explainer = Explainer(
            model=wrapper,
            algorithm=GNNExplainer(
                epochs=self.gnn_explainer_epochs,
                lr=self.gnn_explainer_lr,
            ),
            explanation_type="model",
            model_config=ModelConfig(
                mode="binary_classification",
                task_level="edge",
                return_type="probs",
            ),
            node_mask_type="attributes",
            edge_mask_type="object",
        )

        try:
            return explainer(
                x={node_type: graph[node_type].x for node_type in graph.node_types},
                edge_index={relation: graph[relation].edge_index for relation in graph.edge_types},
                edge_attr={relation: graph[relation].edge_attr for relation in graph.edge_types},
                index=edge_index,
            )
        except Exception:
            return None

    def _gradient_feature_importance(
        self,
        graph: HeteroData,
        *,
        edge_type: tuple[str, str, str],
        edge_index: int,
    ) -> dict[str, float]:
        self.model.zero_grad(set_to_none=True)
        clone = _clone_graph_for_gradients(graph)
        outputs = self.model(clone, return_logits=True)
        selected_logit = outputs[edge_type][edge_index].reshape(())
        selected_logit.backward()

        edge_store = clone[edge_type]
        edge_attr_grad = edge_store.edge_attr.grad[edge_index].detach().abs()
        src_node_index = int(edge_store.edge_index[0, edge_index].item())
        dst_node_index = int(edge_store.edge_index[1, edge_index].item())
        src_grad = clone[edge_type[0]].x.grad[src_node_index].detach().abs()
        dst_grad = clone[edge_type[2]].x.grad[dst_node_index].detach().abs()
        node_grad = (src_grad + dst_grad) / 2.0

        contributions: dict[str, float] = {}
        for feature_name, value in zip(EDGE_FEATURE_COLUMNS, edge_attr_grad.tolist(), strict=True):
            contributions[EDGE_FEATURE_LABELS[feature_name]] = float(value)
        for feature_name, value in zip(NODE_FEATURE_COLUMNS, node_grad.tolist(), strict=True):
            contributions[NODE_FEATURE_LABELS[feature_name]] = float(value)
        return contributions

    def _top_features(self, contributions: dict[str, float]) -> list[FeatureContribution]:
        if not contributions:
            return []

        total = sum(contributions.values())
        if total <= 0.0:
            uniform = 1.0 / len(contributions)
            return [
                FeatureContribution(feature=name, weight=uniform)
                for name in list(contributions.keys())[:5]
            ]

        ranked = sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:5]
        return [
            FeatureContribution(feature=name, weight=value / total)
            for name, value in ranked
        ]

    def _critical_edges(
        self,
        *,
        graph: HeteroData,
        edge_type: tuple[str, str, str],
        edge_index: int,
        explanation: Any | None,
    ) -> list[CriticalEdge]:
        relation_masks: list[tuple[tuple[str, str, str], Tensor]] = []
        if explanation is not None:
            for relation in graph.edge_types:
                try:
                    edge_mask = explanation[relation].edge_mask
                except Exception:
                    edge_mask = None
                if edge_mask is not None and edge_mask.numel() > 0:
                    relation_masks.append((relation, edge_mask.detach().cpu()))

        if not relation_masks:
            relation_masks = [(edge_type, torch.ones(graph[edge_type].edge_index.shape[1]))]

        critical: list[CriticalEdge] = []
        for relation, mask in relation_masks:
            edge_store = graph[relation]
            top_k = min(2, mask.numel())
            if top_k == 0:
                continue
            values, indices = torch.topk(mask, k=top_k)
            for value, index in zip(values.tolist(), indices.tolist(), strict=True):
                source_index = int(edge_store.edge_index[0, index].item())
                destination_index = int(edge_store.edge_index[1, index].item())
                source = graph[relation[0]].upi_id[source_index]
                destination = graph[relation[2]].upi_id[destination_index]
                flag = self._edge_flag(edge_store.edge_attr[index])
                critical.append(
                    CriticalEdge(
                        source=source,
                        destination=destination,
                        weight=float(value),
                        flag=flag,
                    )
                )
        critical.sort(key=lambda item: item.weight, reverse=True)
        return critical[:2]

    def _edge_flag(self, edge_attr: Tensor) -> str:
        delta_t = float(edge_attr[1].item())
        temporal_decay = float(edge_attr[2].item())
        cycle_flag = float(edge_attr[3].item())
        mcc_weight = float(edge_attr[4].item())

        if cycle_flag >= 0.5:
            return "cycle"
        if temporal_decay >= 0.8 or delta_t <= 120.0:
            return "velocity"
        if mcc_weight >= 1.5:
            return "merchant_risk"
        return "normal"

    def _fraud_pattern(
        self,
        *,
        graph: HeteroData,
        edge_type: tuple[str, str, str],
        edge_index: int,
        critical_edges: list[CriticalEdge],
    ) -> str:
        edge_store = graph[edge_type]
        source_index = int(edge_store.edge_index[0, edge_index].item())
        destination_index = int(edge_store.edge_index[1, edge_index].item())

        incoming_to_destination = int((edge_store.edge_index[1] == destination_index).sum().item())
        outgoing_from_source = int((edge_store.edge_index[0] == source_index).sum().item())
        if any(edge.flag == "cycle" for edge in critical_edges):
            return "Triangular cycle pattern with circular routing"
        if incoming_to_destination >= 3:
            return f"Star topology - central node aggregating from {incoming_to_destination} sources"
        if outgoing_from_source >= 3:
            return f"Fan-out pattern - source dispersing across {outgoing_from_source} destinations"
        return "High-risk merchant routing with elevated temporal velocity"

    def _analyst_summary(
        self,
        *,
        graph: HeteroData,
        edge_type: tuple[str, str, str],
        edge_index: int,
        risk_score: float,
        top_features: list[FeatureContribution],
        critical_edges: list[CriticalEdge],
        fraud_pattern: str,
    ) -> str:
        edge_attr = graph[edge_type].edge_attr[edge_index]
        amount_indicator = math.exp(float(edge_attr[0].item())) - 1.0
        feature_text = ", ".join(item.feature for item in top_features[:3]) or "graph context"
        edge_flags = ", ".join(edge.flag for edge in critical_edges) or "none"
        return (
            f"Transaction scored {risk_score:.2f} due to {feature_text}. "
            f"Observed pattern: {fraud_pattern}. "
            f"Key edge signals were {edge_flags}; approximate transaction amount signal was {amount_indicator:.0f} INR."
        )


def _clone_graph_for_gradients(graph: HeteroData) -> HeteroData:
    clone = HeteroData()
    for node_type in graph.node_types:
        clone[node_type].x = graph[node_type].x.detach().clone().float().requires_grad_(True)
        clone[node_type].num_nodes = graph[node_type].num_nodes
        clone[node_type].upi_id = list(getattr(graph[node_type], "upi_id", []))
        if hasattr(graph[node_type], "synthetic_mask"):
            clone[node_type].synthetic_mask = graph[node_type].synthetic_mask.clone()

    for edge_type in graph.edge_types:
        clone[edge_type].edge_index = graph[edge_type].edge_index.clone()
        clone[edge_type].edge_attr = (
            graph[edge_type].edge_attr.detach().clone().float().requires_grad_(True)
        )
        if hasattr(graph[edge_type], "y"):
            clone[edge_type].y = graph[edge_type].y.clone()
        if hasattr(graph[edge_type], "timestamp"):
            clone[edge_type].timestamp = graph[edge_type].timestamp.clone()
        clone[edge_type].txn_id = list(getattr(graph[edge_type], "txn_id", []))
    return clone
