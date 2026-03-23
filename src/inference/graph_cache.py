"""Graph cache for real-time inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.data.paysim_loader import load_config

try:
    from redis import Redis
except ImportError:  # pragma: no cover
    Redis = None  # type: ignore


@dataclass
class EmbeddingCacheEntry:
    embedding: Tensor
    expires_at: float


class GraphCache:
    """Retrieve temporal subgraphs and cache user embeddings."""

    def __init__(
        self,
        *,
        graph: HeteroData | None = None,
        ttl_seconds: int = 300,
        redis_url: str | None = None,
        config_path: str | None = None,
    ) -> None:
        self.config = load_config(config_path)
        self.ttl_seconds = ttl_seconds
        self.graph = graph
        self.redis_url = redis_url
        self.redis_client = Redis.from_url(redis_url) if redis_url and Redis is not None else None
        self.embedding_cache: dict[str, EmbeddingCacheEntry] = {}

    def load_graph(self, graph: HeteroData) -> None:
        self.graph = graph

    def get_temporal_subgraph(
        self,
        *,
        src_upi: str,
        timestamp: datetime | pd.Timestamp | str,
        lookback_hours: int = 24,
        hops: int = 2,
    ) -> HeteroData:
        """Return a temporal 2-hop neighborhood for the source user."""

        if self.graph is None:
            return self.empty_subgraph()

        resolved_timestamp = int(pd.Timestamp(timestamp).timestamp())
        cutoff = resolved_timestamp - int(lookback_hours * 3600)
        temporal_adjacency, filtered_edges = self._temporal_adjacency(
            cutoff_timestamp=cutoff,
            upper_timestamp=resolved_timestamp,
        )

        user_nodes: set[str] = set()
        merchant_nodes: set[str] = set()
        frontier: list[tuple[str, str]] = []

        if src_upi in set(self.graph["user"].upi_id):
            user_nodes.add(src_upi)
            frontier.append(("user", src_upi))

        for _ in range(max(0, hops)):
            next_frontier: list[tuple[str, str]] = []
            for node_type, upi_id in frontier:
                for neighbor_type, neighbor_id in temporal_adjacency.get((node_type, upi_id), set()):
                    if neighbor_type == "user" and neighbor_id not in user_nodes:
                        user_nodes.add(neighbor_id)
                        next_frontier.append((neighbor_type, neighbor_id))
                    if neighbor_type == "merchant" and neighbor_id not in merchant_nodes:
                        merchant_nodes.add(neighbor_id)
                        next_frontier.append((neighbor_type, neighbor_id))
            frontier = next_frontier

        subgraph = self._extract_subgraph(
            user_ids=user_nodes,
            merchant_ids=merchant_nodes,
            filtered_edges=filtered_edges,
        )
        if sum(subgraph[edge_type].edge_index.shape[1] for edge_type in subgraph.edge_types) == 0 and src_upi in set(self.graph["user"].upi_id):
            user_nodes, merchant_nodes = self._expand_neighborhood_without_time_filter(
                src_upi=src_upi,
                hops=hops,
            )
            fallback_edges = {
                relation: torch.ones(self.graph[relation].edge_index.shape[1], dtype=torch.bool)
                for relation in self.graph.edge_types
            }
            return self._extract_subgraph(
                user_ids=user_nodes,
                merchant_ids=merchant_nodes,
                filtered_edges=fallback_edges,
            )
        return subgraph

    def get_cached_embedding(self, upi_id: str) -> Tensor | None:
        """Return a cached embedding if it is still valid."""

        entry = self.embedding_cache.get(upi_id)
        if entry is None:
            return None
        if entry.expires_at < time.time():
            self.embedding_cache.pop(upi_id, None)
            return None
        return entry.embedding.clone()

    def set_cached_embedding(self, upi_id: str, embedding: Tensor) -> None:
        """Store an embedding with TTL semantics."""

        self.embedding_cache[upi_id] = EmbeddingCacheEntry(
            embedding=embedding.detach().cpu().clone(),
            expires_at=time.time() + self.ttl_seconds,
        )

    def empty_subgraph(self) -> HeteroData:
        """Create an empty graph shell with the expected feature dimensions."""

        relations = self._relations()
        user_dim = self.graph["user"].x.shape[1] if self.graph is not None else 5
        merchant_dim = self.graph["merchant"].x.shape[1] if self.graph is not None else 5
        edge_dim = 5

        empty = HeteroData()
        empty["user"].x = torch.empty((0, user_dim), dtype=torch.float32)
        empty["user"].num_nodes = 0
        empty["user"].upi_id = []
        empty["merchant"].x = torch.empty((0, merchant_dim), dtype=torch.float32)
        empty["merchant"].num_nodes = 0
        empty["merchant"].upi_id = []

        for relation in relations:
            empty[relation].edge_index = torch.empty((2, 0), dtype=torch.long)
            empty[relation].edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)
            empty[relation].y = torch.empty((0,), dtype=torch.float32)
            empty[relation].timestamp = torch.empty((0,), dtype=torch.long)
            empty[relation].txn_id = []
        return empty

    def _temporal_adjacency(
        self,
        *,
        cutoff_timestamp: int,
        upper_timestamp: int,
    ) -> tuple[dict[tuple[str, str], set[tuple[str, str]]], dict[tuple[str, str, str], Tensor]]:
        adjacency: dict[tuple[str, str], set[tuple[str, str]]] = {}
        filtered_edges: dict[tuple[str, str, str], Tensor] = {}
        if self.graph is None:
            return adjacency, filtered_edges

        for relation in self.graph.edge_types:
            edge_store = self.graph[relation]
            if edge_store.edge_index.numel() == 0:
                filtered_edges[relation] = torch.zeros(0, dtype=torch.bool)
                continue

            time_mask = (edge_store.timestamp >= cutoff_timestamp) & (edge_store.timestamp <= upper_timestamp)
            filtered_edges[relation] = time_mask

            for src_index, dst_index in edge_store.edge_index[:, time_mask].t().tolist():
                src_upi = self.graph[relation[0]].upi_id[int(src_index)]
                dst_upi = self.graph[relation[2]].upi_id[int(dst_index)]
                adjacency.setdefault((relation[0], src_upi), set()).add((relation[2], dst_upi))
                adjacency.setdefault((relation[2], dst_upi), set()).add((relation[0], src_upi))

        return adjacency, filtered_edges

    def _extract_subgraph(
        self,
        *,
        user_ids: set[str],
        merchant_ids: set[str],
        filtered_edges: dict[tuple[str, str, str], Tensor],
    ) -> HeteroData:
        if self.graph is None:
            return self.empty_subgraph()

        if not user_ids and not merchant_ids:
            return self.empty_subgraph()

        subgraph = HeteroData()
        user_indices = self._indices_for_ids("user", user_ids)
        merchant_indices = self._indices_for_ids("merchant", merchant_ids)
        user_map = self._index_map(self.graph["user"].num_nodes, user_indices)
        merchant_map = self._index_map(self.graph["merchant"].num_nodes, merchant_indices)

        subgraph["user"].x = self.graph["user"].x[user_indices].clone()
        subgraph["user"].num_nodes = int(user_indices.numel())
        subgraph["user"].upi_id = [self.graph["user"].upi_id[index] for index in user_indices.tolist()]
        if hasattr(self.graph["user"], "synthetic_mask"):
            subgraph["user"].synthetic_mask = self.graph["user"].synthetic_mask[user_indices].clone()

        subgraph["merchant"].x = self.graph["merchant"].x[merchant_indices].clone()
        subgraph["merchant"].num_nodes = int(merchant_indices.numel())
        subgraph["merchant"].upi_id = [
            self.graph["merchant"].upi_id[index] for index in merchant_indices.tolist()
        ]

        for relation in self._relations():
            source_map = user_map
            destination_map = user_map if relation[2] == "user" else merchant_map
            edge_store = self.graph[relation]
            base_mask = filtered_edges.get(relation, torch.zeros(edge_store.edge_index.shape[1], dtype=torch.bool))
            src_local = source_map[edge_store.edge_index[0]]
            dst_local = destination_map[edge_store.edge_index[1]]
            keep_mask = base_mask & (src_local >= 0) & (dst_local >= 0)

            if not keep_mask.any():
                subgraph[relation].edge_index = torch.empty((2, 0), dtype=torch.long)
                subgraph[relation].edge_attr = torch.empty(
                    (0, edge_store.edge_attr.shape[1]),
                    dtype=edge_store.edge_attr.dtype,
                )
                subgraph[relation].y = torch.empty((0,), dtype=edge_store.y.dtype)
                subgraph[relation].timestamp = torch.empty((0,), dtype=edge_store.timestamp.dtype)
                subgraph[relation].txn_id = []
                continue

            kept_indices = torch.nonzero(keep_mask, as_tuple=False).view(-1)
            subgraph[relation].edge_index = torch.stack(
                [src_local[keep_mask], dst_local[keep_mask]],
                dim=0,
            )
            subgraph[relation].edge_attr = edge_store.edge_attr[keep_mask].clone()
            subgraph[relation].y = edge_store.y[keep_mask].clone()
            subgraph[relation].timestamp = edge_store.timestamp[keep_mask].clone()
            subgraph[relation].txn_id = [edge_store.txn_id[index] for index in kept_indices.tolist()]

        subgraph.validate(raise_on_error=True)
        return subgraph

    def _indices_for_ids(self, node_type: str, upi_ids: set[str]) -> Tensor:
        if self.graph is None or not upi_ids:
            return torch.empty((0,), dtype=torch.long)
        indices = [
            index
            for index, upi_id in enumerate(self.graph[node_type].upi_id)
            if upi_id in upi_ids
        ]
        return torch.tensor(indices, dtype=torch.long)

    def _index_map(self, total_size: int, kept_indices: Tensor) -> Tensor:
        index_map = torch.full((int(total_size),), -1, dtype=torch.long)
        if kept_indices.numel():
            index_map[kept_indices] = torch.arange(kept_indices.numel(), dtype=torch.long)
        return index_map

    def _relations(self) -> list[tuple[str, str, str]]:
        if self.graph is not None:
            return list(self.graph.edge_types)
        relation_names = self.config["graph"]["relations"]
        return [
            ("user", relation_names["p2p"], "user"),
            ("user", relation_names["merchant"], "merchant"),
        ]

    def _expand_neighborhood_without_time_filter(
        self,
        *,
        src_upi: str,
        hops: int,
    ) -> tuple[set[str], set[str]]:
        user_nodes: set[str] = set()
        merchant_nodes: set[str] = set()
        if self.graph is None:
            return user_nodes, merchant_nodes

        adjacency: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for relation in self.graph.edge_types:
            edge_store = self.graph[relation]
            for src_index, dst_index in edge_store.edge_index.t().tolist():
                source = self.graph[relation[0]].upi_id[int(src_index)]
                destination = self.graph[relation[2]].upi_id[int(dst_index)]
                adjacency.setdefault((relation[0], source), set()).add((relation[2], destination))
                adjacency.setdefault((relation[2], destination), set()).add((relation[0], source))

        frontier = [("user", src_upi)]
        user_nodes.add(src_upi)
        for _ in range(max(0, hops)):
            next_frontier: list[tuple[str, str]] = []
            for node_type, upi_id in frontier:
                for neighbor_type, neighbor_id in adjacency.get((node_type, upi_id), set()):
                    if neighbor_type == "user" and neighbor_id not in user_nodes:
                        user_nodes.add(neighbor_id)
                        next_frontier.append((neighbor_type, neighbor_id))
                    if neighbor_type == "merchant" and neighbor_id not in merchant_nodes:
                        merchant_nodes.add(neighbor_id)
                        next_frontier.append((neighbor_type, neighbor_id))
            frontier = next_frontier
        return user_nodes, merchant_nodes
