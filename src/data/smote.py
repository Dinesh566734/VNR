"""Graph-SMOTE style augmentation for the training graph."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.data import HeteroData

from .paysim_loader import PROJECT_ROOT, load_config


@dataclass(frozen=True)
class CandidateEdge:
    relation: tuple[str, str, str]
    destination_index: int
    score: float


class EdgeGeneratorMLP(nn.Module):
    """Small MLP used to predict whether a synthetic edge should exist."""

    def __init__(self, input_dim: int, hidden_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def apply_graph_smote(
    train_graph: HeteroData,
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    persist: bool = True,
) -> HeteroData:
    """
    Augment the training graph until the fraud edge rate reaches the target band.

    The current graph labels transactions on edges rather than nodes, so the
    minority seed set is derived from user nodes touched by fraudulent edges.
    Synthetic nodes are generated in user space and connected back to the graph
    through a learned edge generator over existing node features.
    """

    config = load_config(config_path)
    smote_config = config["smote"]
    augmented = copy.deepcopy(train_graph)
    _ensure_user_metadata(augmented)

    seed_users = _fraud_seed_users(augmented)
    if not seed_users:
        raise ValueError("Graph-SMOTE requires at least one fraud seed user node.")

    user_features = augmented["user"].x.detach().clone().float()
    if user_features.shape[0] <= 1:
        raise ValueError("Graph-SMOTE requires at least two user nodes.")

    k_neighbors = min(int(smote_config["k_neighbors"]), max(user_features.shape[0] - 1, 1))
    nearest_neighbors = _nearest_neighbors(
        user_features,
        seed_indices=seed_users,
        k=k_neighbors,
    )
    edge_generator = _train_edge_generator(augmented, config=config)
    relation_defaults = _relation_default_edge_attr(augmented)
    rng = torch.Generator().manual_seed(int(config["data"]["random_seed"]))

    target_rate = float(smote_config["target_fraud_rate"])
    max_rate = float(smote_config["max_fraud_rate"])
    min_edges = int(smote_config["min_edges_per_synthetic"])
    max_edges = int(smote_config["max_edges_per_synthetic"])
    max_new_nodes = max(
        int(smote_config["max_synthetic_nodes"]),
        _required_synthetic_node_budget(
            graph=augmented,
            target_rate=target_rate,
            min_edges_per_synthetic=min_edges,
        ),
    )

    original_max_timestamp = _max_edge_timestamp(augmented)
    iterations = 0
    while _fraud_rate(augmented) < target_rate and iterations < max_new_nodes:
        seed_user = seed_users[iterations % len(seed_users)]
        neighbor_candidates = nearest_neighbors[seed_user]
        neighbor_user = int(neighbor_candidates[iterations % len(neighbor_candidates)])
        interpolation = float(torch.rand(1, generator=rng).item())
        synthetic_feature = user_features[seed_user] + interpolation * (
            user_features[neighbor_user] - user_features[seed_user]
        )

        synthetic_user_index = _append_synthetic_user(augmented, synthetic_feature)
        user_features = augmented["user"].x.detach().clone().float()
        candidate_edges = _candidate_edges_for_synthetic_user(
            graph=augmented,
            edge_generator=edge_generator,
            synthetic_feature=synthetic_feature,
            seed_user=seed_user,
            neighbor_user=neighbor_user,
            threshold=float(smote_config["edge_generator"]["threshold"]),
            max_edges=max_edges,
        )
        if len(candidate_edges) < min_edges:
            candidate_edges = _force_minimum_candidates(
                graph=augmented,
                edge_generator=edge_generator,
                synthetic_feature=synthetic_feature,
                seed_user=seed_user,
                neighbor_user=neighbor_user,
                minimum=min_edges,
            )

        for edge_offset, candidate in enumerate(candidate_edges[:max_edges]):
            edge_attr = _synthetic_edge_attr(
                graph=augmented,
                relation=candidate.relation,
                seed_user=seed_user,
                neighbor_user=neighbor_user,
                interpolation=interpolation,
                defaults=relation_defaults,
            )
            _append_synthetic_edge(
                graph=augmented,
                relation=candidate.relation,
                source_index=synthetic_user_index,
                destination_index=candidate.destination_index,
                edge_attr=edge_attr,
                timestamp=original_max_timestamp + iterations + edge_offset + 1,
                txn_id=f"SMOTE_TXN_{iterations:05d}_{edge_offset:02d}",
            )

        iterations += 1
        if _fraud_rate(augmented) >= max_rate:
            break

    final_rate = _fraud_rate(augmented)
    if not float(smote_config["min_fraud_rate"]) <= final_rate <= max_rate:
        raise AssertionError(
            f"Augmented fraud rate {final_rate:.4f} is outside the configured SMOTE band."
        )

    augmented.smote_metadata = {
        "synthetic_user_count": int(augmented["user"].synthetic_mask.sum().item()),
        "fraud_rate": float(final_rate),
        "target_fraud_rate": float(target_rate),
    }

    if persist:
        resolved_output = (
            Path(output_path)
            if output_path is not None
            else PROJECT_ROOT / config["data"]["processed_dir"] / "graph_train_smote.pt"
        )
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(augmented, resolved_output)

    return augmented


def _train_edge_generator(
    graph: HeteroData,
    config: dict[str, Any],
) -> EdgeGeneratorMLP:
    user_dim = int(graph["user"].x.shape[1])
    hidden_channels = int(config["smote"]["edge_generator"]["hidden_channels"])
    learning_rate = float(config["smote"]["edge_generator"]["learning_rate"])
    epochs = int(config["smote"]["edge_generator"]["epochs"])
    generator = torch.Generator().manual_seed(int(config["data"]["random_seed"]))

    model = EdgeGeneratorMLP(input_dim=user_dim * 2, hidden_channels=hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    features, labels = _edge_generator_dataset(graph=graph, generator=generator)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    return model.eval()


def _edge_generator_dataset(
    graph: HeteroData,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    positives: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for relation in graph.edge_types:
        edge_store = graph[relation]
        destination_type = relation[2]
        if edge_store.edge_index.numel() == 0:
            continue

        src_features = graph["user"].x[edge_store.edge_index[0]]
        dst_features = graph[destination_type].x[edge_store.edge_index[1]]
        positives.append(torch.cat([src_features, dst_features], dim=1))
        labels.append(torch.ones(edge_store.edge_index.shape[1], dtype=torch.float32))

        negative_features = _negative_samples_for_relation(
            graph=graph,
            relation=relation,
            sample_count=edge_store.edge_index.shape[1],
            generator=generator,
        )
        positives.append(negative_features)
        labels.append(torch.zeros(negative_features.shape[0], dtype=torch.float32))

    return torch.cat(positives, dim=0), torch.cat(labels, dim=0)


def _negative_samples_for_relation(
    graph: HeteroData,
    relation: tuple[str, str, str],
    sample_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    destination_type = relation[2]
    existing = {
        (int(src), int(dst))
        for src, dst in zip(
            graph[relation].edge_index[0].tolist(),
            graph[relation].edge_index[1].tolist(),
        )
    }
    negatives: list[torch.Tensor] = []
    max_attempts = sample_count * 10 + 1
    attempts = 0

    while len(negatives) < sample_count and attempts < max_attempts:
        attempts += 1
        src_index = int(
            torch.randint(graph["user"].num_nodes, (1,), generator=generator).item()
        )
        dst_index = int(
            torch.randint(graph[destination_type].num_nodes, (1,), generator=generator).item()
        )
        if (src_index, dst_index) in existing:
            continue
        negatives.append(
            torch.cat([graph["user"].x[src_index], graph[destination_type].x[dst_index]])
        )

    if not negatives:
        return torch.empty((0, graph["user"].x.shape[1] * 2), dtype=torch.float32)

    return torch.stack(negatives).float()


def _fraud_seed_users(graph: HeteroData) -> list[int]:
    seeds: set[int] = set()
    for relation in graph.edge_types:
        edge_store = graph[relation]
        if edge_store.y.numel() == 0:
            continue
        fraud_mask = edge_store.y > 0.5
        if fraud_mask.any():
            seeds.update(edge_store.edge_index[0, fraud_mask].tolist())
            if relation[2] == "user":
                seeds.update(edge_store.edge_index[1, fraud_mask].tolist())
    return sorted(int(seed) for seed in seeds)


def _required_synthetic_node_budget(
    *,
    graph: HeteroData,
    target_rate: float,
    min_edges_per_synthetic: int,
) -> int:
    """Estimate how many synthetic users are required to reach the target fraud rate."""

    total_edges = sum(int(graph[relation].y.numel()) for relation in graph.edge_types)
    fraud_edges = sum(int((graph[relation].y > 0.5).sum().item()) for relation in graph.edge_types)
    if total_edges == 0 or fraud_edges / total_edges >= float(target_rate):
        return 0

    added_fraud_edges = (float(target_rate) * total_edges - fraud_edges) / (1.0 - float(target_rate))
    required_edges = max(0, int(torch.ceil(torch.tensor(added_fraud_edges)).item()))
    if min_edges_per_synthetic <= 0:
        return required_edges
    return int((required_edges + min_edges_per_synthetic - 1) // min_edges_per_synthetic)


def _nearest_neighbors(
    features: torch.Tensor,
    *,
    seed_indices: list[int],
    k: int,
) -> dict[int, list[int]]:
    """
    Return k nearest real-user neighbors for each fraud seed user.

    The original implementation built a full all-pairs distance matrix with
    `torch.cdist`, which becomes quadratic in memory and fails on realistic
    PaySim slices. We only ever query neighbors for fraud seed users, so fit
    once on all real users and ask for the small subset we actually need.
    """

    if features.shape[0] <= 1:
        return {int(seed): [] for seed in seed_indices}

    all_candidates = list(range(int(features.shape[0])))
    neighbor_map: dict[int, list[int]] = {}
    neighbor_count = min(int(k) + 1, int(features.shape[0]))
    batch_size = 256
    all_features = features.detach().cpu().float()

    for start in range(0, len(seed_indices), batch_size):
        batch_seed_indices = seed_indices[start : start + batch_size]
        seed_tensor = torch.tensor(batch_seed_indices, dtype=torch.long)
        seed_features = all_features[seed_tensor]
        distances = torch.cdist(seed_features, all_features, p=2)
        row_indices = torch.arange(len(batch_seed_indices), dtype=torch.long)
        distances[row_indices, seed_tensor] = float("inf")
        indices = torch.topk(distances, k=neighbor_count, largest=False).indices

        for seed_user, neighbor_row in zip(batch_seed_indices, indices.tolist()):
            resolved = [int(index) for index in neighbor_row if int(index) != int(seed_user)]
            if len(resolved) < int(k):
                resolved.extend(
                    candidate
                    for candidate in all_candidates
                    if candidate != int(seed_user) and candidate not in resolved
                )
            neighbor_map[int(seed_user)] = resolved[: int(k)]

    return neighbor_map


def _candidate_edges_for_synthetic_user(
    graph: HeteroData,
    edge_generator: EdgeGeneratorMLP,
    synthetic_feature: torch.Tensor,
    seed_user: int,
    neighbor_user: int,
    threshold: float,
    max_edges: int,
) -> list[CandidateEdge]:
    candidate_pool = _candidate_destination_pool(graph, seed_user=seed_user, neighbor_user=neighbor_user)
    scored: list[CandidateEdge] = []

    with torch.no_grad():
        for relation, destinations in candidate_pool.items():
            destination_type = relation[2]
            for destination_index in sorted(destinations):
                input_vector = torch.cat(
                    [synthetic_feature, graph[destination_type].x[destination_index]]
                ).unsqueeze(0)
                probability = torch.sigmoid(edge_generator(input_vector)).item()
                if probability >= threshold:
                    scored.append(
                        CandidateEdge(
                            relation=relation,
                            destination_index=int(destination_index),
                            score=float(probability),
                        )
                    )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:max_edges]


def _force_minimum_candidates(
    graph: HeteroData,
    edge_generator: EdgeGeneratorMLP,
    synthetic_feature: torch.Tensor,
    seed_user: int,
    neighbor_user: int,
    minimum: int,
) -> list[CandidateEdge]:
    candidate_pool = _candidate_destination_pool(graph, seed_user=seed_user, neighbor_user=neighbor_user)
    scored: list[CandidateEdge] = []

    with torch.no_grad():
        for relation, destinations in candidate_pool.items():
            destination_type = relation[2]
            for destination_index in sorted(destinations):
                input_vector = torch.cat(
                    [synthetic_feature, graph[destination_type].x[destination_index]]
                ).unsqueeze(0)
                probability = torch.sigmoid(edge_generator(input_vector)).item()
                scored.append(
                    CandidateEdge(
                        relation=relation,
                        destination_index=int(destination_index),
                        score=float(probability),
                    )
                )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:minimum]


def _candidate_destination_pool(
    graph: HeteroData,
    seed_user: int,
    neighbor_user: int,
) -> dict[tuple[str, str, str], set[int]]:
    pool: dict[tuple[str, str, str], set[int]] = {}
    for relation in graph.edge_types:
        edge_store = graph[relation]
        source_mask = (edge_store.edge_index[0] == seed_user) | (edge_store.edge_index[0] == neighbor_user)
        destinations = edge_store.edge_index[1, source_mask].tolist()
        if destinations:
            pool[relation] = set(int(destination) for destination in destinations)

    if not pool:
        for relation in graph.edge_types:
            if graph[relation].edge_index.numel():
                pool[relation] = set(
                    int(destination) for destination in graph[relation].edge_index[1].tolist()[:4]
                )
    return pool


def _synthetic_edge_attr(
    graph: HeteroData,
    relation: tuple[str, str, str],
    seed_user: int,
    neighbor_user: int,
    interpolation: float,
    defaults: dict[tuple[str, str, str], torch.Tensor],
) -> torch.Tensor:
    edge_store = graph[relation]
    seed_mask = edge_store.edge_index[0] == seed_user
    neighbor_mask = edge_store.edge_index[0] == neighbor_user

    seed_attr = (
        edge_store.edge_attr[seed_mask].mean(dim=0)
        if seed_mask.any()
        else defaults[relation]
    )
    neighbor_attr = (
        edge_store.edge_attr[neighbor_mask].mean(dim=0)
        if neighbor_mask.any()
        else defaults[relation]
    )
    synthetic = seed_attr + interpolation * (neighbor_attr - seed_attr)
    synthetic = synthetic.clone().float()
    synthetic[0] = synthetic[0].clamp(min=0.0)
    synthetic[1] = synthetic[1].clamp(min=0.0)
    synthetic[2] = synthetic[2].clamp(min=0.0, max=1.0)
    synthetic[3] = torch.round(synthetic[3].clamp(min=0.0, max=1.0))
    synthetic[4] = synthetic[4].clamp(min=0.5, max=2.0)
    return synthetic


def _relation_default_edge_attr(
    graph: HeteroData,
) -> dict[tuple[str, str, str], torch.Tensor]:
    defaults: dict[tuple[str, str, str], torch.Tensor] = {}
    for relation in graph.edge_types:
        edge_store = graph[relation]
        if edge_store.edge_attr.numel() == 0:
            defaults[relation] = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
            continue
        fraud_mask = edge_store.y > 0.5
        defaults[relation] = (
            edge_store.edge_attr[fraud_mask].mean(dim=0)
            if fraud_mask.any()
            else edge_store.edge_attr.mean(dim=0)
        ).float()
    return defaults


def _append_synthetic_user(graph: HeteroData, synthetic_feature: torch.Tensor) -> int:
    original_count = int(graph["user"].num_nodes)
    graph["user"].x = torch.cat([graph["user"].x, synthetic_feature.unsqueeze(0)], dim=0)
    graph["user"].num_nodes = graph["user"].x.shape[0]
    graph["user"].upi_id = list(graph["user"].upi_id) + [
        f"synthetic_user_{original_count:05d}"
    ]
    graph["user"].synthetic_mask = torch.cat(
        [graph["user"].synthetic_mask, torch.tensor([True], dtype=torch.bool)]
    )
    return original_count


def _append_synthetic_edge(
    graph: HeteroData,
    relation: tuple[str, str, str],
    source_index: int,
    destination_index: int,
    edge_attr: torch.Tensor,
    timestamp: int,
    txn_id: str,
) -> None:
    edge_store = graph[relation]
    new_edge_index = torch.tensor([[source_index], [destination_index]], dtype=torch.long)
    edge_store.edge_index = torch.cat([edge_store.edge_index, new_edge_index], dim=1)
    edge_store.edge_attr = torch.cat([edge_store.edge_attr, edge_attr.unsqueeze(0)], dim=0)
    edge_store.y = torch.cat([edge_store.y, torch.tensor([1.0], dtype=torch.float32)])
    edge_store.timestamp = torch.cat([edge_store.timestamp, torch.tensor([timestamp], dtype=torch.long)])
    edge_store.txn_id = list(edge_store.txn_id) + [txn_id]


def _ensure_user_metadata(graph: HeteroData) -> None:
    if not hasattr(graph["user"], "synthetic_mask"):
        graph["user"].synthetic_mask = torch.zeros(graph["user"].num_nodes, dtype=torch.bool)


def _fraud_rate(graph: HeteroData) -> float:
    labels = [graph[relation].y.float() for relation in graph.edge_types if graph[relation].y.numel()]
    if not labels:
        return 0.0
    combined = torch.cat(labels, dim=0)
    return float(combined.mean().item())


def _max_edge_timestamp(graph: HeteroData) -> int:
    timestamps = [
        graph[relation].timestamp.max().item()
        for relation in graph.edge_types
        if graph[relation].timestamp.numel()
    ]
    return int(max(timestamps, default=0))
