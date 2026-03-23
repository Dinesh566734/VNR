"""Partition the training graph into mini-batch communities."""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import HeteroData

from .paysim_loader import PROJECT_ROOT, load_config

try:
    import pymetis  # type: ignore
except ImportError:  # pragma: no cover - exercised indirectly through fallback
    pymetis = None


def partition_training_graph(
    train_graph: HeteroData,
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """
    Partition a training graph into cluster assignments.

    When `pymetis` is available it is used directly. Some Windows environments
    expose a 32-bit MinGW compiler ahead of the 64-bit Python toolchain, which
    causes `pymetis` builds to fail during installation. In those cases the
    function falls back to a deterministic neighborhood-preserving partitioner.
    """

    config = load_config(config_path)
    cluster_count = int(config["partitioner"]["num_clusters"])
    adjacency, node_lookup = _combined_undirected_adjacency(train_graph)
    if len(node_lookup) < cluster_count:
        raise ValueError(
            f"Graph has {len(node_lookup)} nodes, fewer than requested {cluster_count} clusters."
        )

    if pymetis is not None:
        _, membership = pymetis.part_graph(cluster_count, adjacency=adjacency)
        method = "pymetis"
    else:
        membership = _fallback_partition(adjacency=adjacency, num_clusters=cluster_count)
        method = "balanced_neighborhood_fallback"

    assignments = _split_assignments_by_type(
        membership=membership,
        node_lookup=node_lookup,
        train_graph=train_graph,
        cluster_count=cluster_count,
    )
    assignments["method"] = method
    assignments["num_clusters"] = cluster_count

    if persist:
        resolved_output = (
            Path(output_path)
            if output_path is not None
            else PROJECT_ROOT / config["data"]["processed_dir"] / config["partitioner"]["assignment_file"]
        )
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output.open("wb") as handle:
            pickle.dump(assignments, handle)

    return assignments


def _combined_undirected_adjacency(
    graph: HeteroData,
) -> tuple[list[list[int]], list[tuple[str, int]]]:
    user_count = int(graph["user"].num_nodes)
    merchant_count = int(graph["merchant"].num_nodes)
    total_nodes = user_count + merchant_count
    adjacency: list[set[int]] = [set() for _ in range(total_nodes)]
    node_lookup = [("user", index) for index in range(user_count)] + [
        ("merchant", index) for index in range(merchant_count)
    ]

    for relation in graph.edge_types:
        edge_store = graph[relation]
        if edge_store.edge_index.numel() == 0:
            continue

        src_offset = 0
        dst_offset = 0 if relation[2] == "user" else user_count
        sources = edge_store.edge_index[0].tolist()
        destinations = edge_store.edge_index[1].tolist()
        for source_index, destination_index in zip(sources, destinations):
            global_source = int(source_index) + src_offset
            global_destination = int(destination_index) + dst_offset
            if global_source == global_destination:
                continue
            adjacency[global_source].add(global_destination)
            adjacency[global_destination].add(global_source)

    return [sorted(neighbors) for neighbors in adjacency], node_lookup


def _fallback_partition(
    adjacency: list[list[int]],
    num_clusters: int,
) -> list[int]:
    node_count = len(adjacency)
    degrees = sorted(
        range(node_count),
        key=lambda node: (len(adjacency[node]), -node),
        reverse=True,
    )
    seeds = degrees[:num_clusters]
    membership = [-1] * node_count
    cluster_sizes = [0] * num_clusters

    for cluster_id, seed in enumerate(seeds):
        membership[seed] = cluster_id
        cluster_sizes[cluster_id] = 1

    remaining = [node for node in degrees if membership[node] == -1]
    while remaining:
        progress = False
        next_remaining: list[int] = []
        for node in remaining:
            neighbor_clusters = [membership[neighbor] for neighbor in adjacency[node] if membership[neighbor] != -1]
            if neighbor_clusters:
                counts = Counter(neighbor_clusters)
                chosen_cluster = min(
                    counts,
                    key=lambda cluster_id: (-counts[cluster_id], cluster_sizes[cluster_id], cluster_id),
                )
                membership[node] = chosen_cluster
                cluster_sizes[chosen_cluster] += 1
                progress = True
            else:
                next_remaining.append(node)

        if progress:
            remaining = next_remaining
            continue

        for node in remaining:
            chosen_cluster = min(range(num_clusters), key=lambda cluster_id: (cluster_sizes[cluster_id], cluster_id))
            membership[node] = chosen_cluster
            cluster_sizes[chosen_cluster] += 1
        break

    return membership


def _split_assignments_by_type(
    membership: list[int],
    node_lookup: list[tuple[str, int]],
    train_graph: HeteroData,
    cluster_count: int,
) -> dict[str, torch.Tensor]:
    user_assignments = torch.full((train_graph["user"].num_nodes,), -1, dtype=torch.long)
    merchant_assignments = torch.full((train_graph["merchant"].num_nodes,), -1, dtype=torch.long)

    for global_index, cluster_id in enumerate(membership):
        node_type, local_index = node_lookup[global_index]
        if node_type == "user":
            user_assignments[local_index] = int(cluster_id)
        else:
            merchant_assignments[local_index] = int(cluster_id)

    used_clusters = torch.cat([user_assignments, merchant_assignments], dim=0).unique()
    if used_clusters.numel() != cluster_count:
        raise AssertionError(
            f"Expected {cluster_count} clusters, but found {used_clusters.numel()} distinct assignments."
        )

    return {"user": user_assignments, "merchant": merchant_assignments}
