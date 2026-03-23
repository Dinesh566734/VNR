"""Build chronological HeteroData graphs from the adapted UPI transactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .feature_engineer import (
    EDGE_FEATURE_COLUMNS,
    NODE_FEATURE_COLUMNS,
    annotate_transactions_with_edge_features,
    build_node_feature_frames,
    prepare_transactions_for_graph,
)
from .paysim_loader import PROJECT_ROOT, load_config


def split_transactions_chronologically(
    transactions: pd.DataFrame,
    config_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Split transactions into train/val/test windows without time leakage."""

    config = load_config(config_path)
    prepared = prepare_transactions_for_graph(transactions)
    ratios = config["graph"]["split_ratios"]
    counts = _split_counts(
        total=len(prepared),
        train_ratio=float(ratios["train"]),
        val_ratio=float(ratios["val"]),
    )

    train_end = _resolve_timestamp_boundary(
        prepared,
        proposed_boundary=counts["train"],
        minimum=1,
        maximum=max(1, len(prepared) - 2),
    )
    val_end = _resolve_timestamp_boundary(
        prepared,
        proposed_boundary=train_end + counts["val"],
        minimum=train_end + 1,
        maximum=max(train_end + 1, len(prepared) - 1),
    )

    splits = {
        "train": prepared.iloc[:train_end].reset_index(drop=True),
        "val": prepared.iloc[train_end:val_end].reset_index(drop=True),
        "test": prepared.iloc[val_end:].reset_index(drop=True),
    }

    _validate_chronological_splits(splits)
    return splits


def build_hetero_graph(
    transactions: pd.DataFrame,
    config_path: str | Path | None = None,
) -> HeteroData:
    """Build a HeteroData graph for a single chronological split."""

    config = load_config(config_path)
    featured = annotate_transactions_with_edge_features(transactions, config_path=config_path)
    node_frames = build_node_feature_frames(featured, config_path=config_path)
    data = HeteroData()

    data["user"].x = _to_feature_tensor(node_frames["user"])
    data["user"].num_nodes = len(node_frames["user"])
    data["user"].upi_id = list(node_frames["user"].index)

    data["merchant"].x = _to_feature_tensor(node_frames["merchant"])
    data["merchant"].num_nodes = len(node_frames["merchant"])
    data["merchant"].upi_id = list(node_frames["merchant"].index)

    user_index = {upi_id: index for index, upi_id in enumerate(node_frames["user"].index)}
    merchant_index = {
        upi_id: index for index, upi_id in enumerate(node_frames["merchant"].index)
    }
    relation_names = config["graph"]["relations"]

    p2p_frame = featured.loc[featured["merchant_type"].eq("peer_to_peer")].reset_index(drop=True)
    merchant_frame = featured.loc[featured["merchant_type"].ne("peer_to_peer")].reset_index(drop=True)

    _attach_relation(
        data=data,
        relation=("user", relation_names["p2p"], "user"),
        edge_frame=p2p_frame,
        src_index=user_index,
        dst_index=user_index,
    )
    _attach_relation(
        data=data,
        relation=("user", relation_names["merchant"], "merchant"),
        edge_frame=merchant_frame,
        src_index=user_index,
        dst_index=merchant_index,
    )

    data.graph_stats = summarize_hetero_graph(data)
    data.validate(raise_on_error=True)
    return data


def build_graph_splits(
    transactions: pd.DataFrame,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    persist: bool = True,
) -> dict[str, HeteroData]:
    """Build, optionally persist, and return train/val/test graph splits."""

    config = load_config(config_path)
    splits = split_transactions_chronologically(transactions, config_path=config_path)
    graphs = {
        split_name: build_hetero_graph(split_frame, config_path=config_path)
        for split_name, split_frame in splits.items()
    }

    if persist:
        target_dir = _resolve_output_dir(config=config, output_dir=output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for split_name, graph in graphs.items():
            filename = config["graph"]["processed_files"][split_name]
            torch.save(graph, target_dir / filename)

    return graphs


def summarize_hetero_graph(graph: HeteroData) -> dict[str, float]:
    """Return the core graph summary metrics requested by the plan."""

    total_nodes = sum(int(graph[node_type].num_nodes) for node_type in graph.node_types)
    total_edges = sum(int(graph[edge_type].edge_index.shape[1]) for edge_type in graph.edge_types)
    labels = [
        graph[edge_type].y
        for edge_type in graph.edge_types
        if hasattr(graph[edge_type], "y") and graph[edge_type].y.numel() > 0
    ]
    all_labels = torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.float32)

    return {
        "num_nodes": float(total_nodes),
        "num_edges": float(total_edges),
        "fraud_rate": float(all_labels.float().mean().item()) if all_labels.numel() else 0.0,
        "node_feature_dim": float(
            max((graph[node_type].x.shape[1] for node_type in graph.node_types), default=0)
        ),
        "edge_feature_dim": float(
            max((graph[edge_type].edge_attr.shape[1] for edge_type in graph.edge_types), default=0)
        ),
        "graph_density": float(
            total_edges / (total_nodes * (total_nodes - 1))
        )
        if total_nodes > 1
        else 0.0,
    }


def build_graphs_from_csv(
    csv_path: str | Path,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, HeteroData]:
    """Convenience wrapper that loads a saved CSV of adapted transactions."""

    frame = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return build_graph_splits(
        transactions=frame,
        config_path=config_path,
        output_dir=output_dir,
        persist=True,
    )


def _attach_relation(
    data: HeteroData,
    relation: tuple[str, str, str],
    edge_frame: pd.DataFrame,
    src_index: dict[str, int],
    dst_index: dict[str, int],
) -> None:
    if edge_frame.empty:
        data[relation].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[relation].edge_attr = torch.empty((0, len(EDGE_FEATURE_COLUMNS)), dtype=torch.float32)
        data[relation].y = torch.empty((0,), dtype=torch.float32)
        data[relation].timestamp = torch.empty((0,), dtype=torch.long)
        data[relation].txn_id = []
        return

    edge_index = torch.from_numpy(
        np.vstack(
            [
                edge_frame["src_upi"].map(src_index).to_numpy(dtype="int64"),
                edge_frame["dst_upi"].map(dst_index).to_numpy(dtype="int64"),
            ]
        )
    )
    edge_attr = torch.tensor(
        edge_frame.loc[:, EDGE_FEATURE_COLUMNS].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    labels = torch.tensor(edge_frame["is_fraud"].to_numpy(dtype="float32"), dtype=torch.float32)
    timestamps = torch.tensor(
        (edge_frame["timestamp"].astype("int64") // 10**9).to_numpy(dtype="int64"),
        dtype=torch.long,
    )

    data[relation].edge_index = edge_index
    data[relation].edge_attr = edge_attr
    data[relation].y = labels
    data[relation].timestamp = timestamps
    data[relation].txn_id = edge_frame["txn_id"].tolist()


def _to_feature_tensor(frame: pd.DataFrame) -> torch.Tensor:
    if frame.empty:
        return torch.empty((0, len(NODE_FEATURE_COLUMNS)), dtype=torch.float32)
    return torch.tensor(
        frame.loc[:, NODE_FEATURE_COLUMNS].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )


def _resolve_output_dir(
    config: dict[str, Any],
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return PROJECT_ROOT / config["data"]["processed_dir"]


def _split_counts(total: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    if total < 3:
        raise ValueError("At least three transactions are required for train/val/test splits.")

    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
        train_count = total - val_count - 1

    return {"train": train_count, "val": val_count, "test": total - train_count - val_count}


def _resolve_timestamp_boundary(
    frame: pd.DataFrame,
    proposed_boundary: int,
    minimum: int,
    maximum: int,
) -> int:
    boundary = min(max(proposed_boundary, minimum), maximum)
    forward = boundary
    while forward < maximum and frame.iloc[forward - 1]["timestamp"] == frame.iloc[forward]["timestamp"]:
        forward += 1
    if forward <= maximum:
        return forward

    backward = boundary
    while backward > minimum and frame.iloc[backward - 1]["timestamp"] == frame.iloc[backward]["timestamp"]:
        backward -= 1
    return backward


def _validate_chronological_splits(splits: dict[str, pd.DataFrame]) -> None:
    train_max = splits["train"]["timestamp"].max()
    val_min = splits["val"]["timestamp"].min()
    val_max = splits["val"]["timestamp"].max()
    test_min = splits["test"]["timestamp"].min()

    if not train_max < val_min:
        raise AssertionError("Training split overlaps validation timestamps.")
    if not val_max < test_min:
        raise AssertionError("Validation split overlaps test timestamps.")
