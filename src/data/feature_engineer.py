"""Node and edge feature engineering for the Sentinel-UPI graph."""

from __future__ import annotations

import hashlib
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .paysim_loader import load_config


NODE_FEATURE_COLUMNS = [
    "account_age",
    "kyc_tier",
    "daily_avg_spend",
    "pagerank_local",
    "betweenness_central",
]
EDGE_FEATURE_COLUMNS = [
    "log_amount",
    "delta_t",
    "temporal_decay",
    "dfs_cycle_flag",
    "mcc_weight",
]


def build_node_feature_frames(
    transactions: pd.DataFrame,
    config_path: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute per-node feature frames for user and merchant node types."""

    config = load_config(config_path)
    prepared = prepare_transactions_for_graph(transactions)
    node_sets = collect_node_ids(prepared)
    interaction_graph = _build_interaction_graph(prepared)
    centrality_scores = _compute_centrality_scores(interaction_graph, config=config)

    return {
        "user": _build_node_feature_frame(
            node_ids=node_sets["user"],
            role="user",
            transactions=prepared,
            centrality_scores=centrality_scores,
            config=config,
        ),
        "merchant": _build_node_feature_frame(
            node_ids=node_sets["merchant"],
            role="merchant",
            transactions=prepared,
            centrality_scores=centrality_scores,
            config=config,
        ),
    }


def annotate_transactions_with_edge_features(
    transactions: pd.DataFrame,
    config_path: str | None = None,
) -> pd.DataFrame:
    """Attach the ordered edge feature columns required by the graph model."""

    config = load_config(config_path)
    prepared = prepare_transactions_for_graph(transactions)
    decay_rate = float(config["features"]["edge"]["temporal_decay_rate"])

    prepared["log_amount"] = np.log1p(prepared["amount_clipped"].clip(lower=0.0))
    prepared["delta_t"] = _compute_source_delta_seconds(prepared)
    prepared["temporal_decay"] = np.exp(-decay_rate * prepared["delta_t"])
    prepared["dfs_cycle_flag"] = compute_cycle_completion_flags(prepared)
    prepared["mcc_weight"] = prepared["mcc_weight"].astype(float)

    return prepared


def prepare_transactions_for_graph(transactions: pd.DataFrame) -> pd.DataFrame:
    """Normalize the transaction frame expected by graph construction."""

    required_columns = {
        "txn_id",
        "src_upi",
        "dst_upi",
        "amount_clipped",
        "timestamp",
        "merchant_type",
        "mcc_weight",
        "is_fraud",
    }
    missing = required_columns.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing graph transaction columns: {sorted(missing)}")

    prepared = transactions.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=False)
    prepared["amount_clipped"] = pd.to_numeric(
        prepared["amount_clipped"], errors="coerce"
    ).fillna(0.0)
    prepared["mcc_weight"] = pd.to_numeric(prepared["mcc_weight"], errors="coerce").fillna(1.0)
    prepared["is_fraud"] = (
        pd.to_numeric(prepared["is_fraud"], errors="coerce").fillna(0).astype(int)
    )
    prepared = prepared.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)
    return prepared


def collect_node_ids(transactions: pd.DataFrame) -> dict[str, pd.Index]:
    """Split account IDs into user and merchant node sets."""

    prepared = prepare_transactions_for_graph(transactions)
    p2p_mask = prepared["merchant_type"].eq("peer_to_peer")
    user_nodes = pd.Index(
        sorted(set(prepared["src_upi"]).union(prepared.loc[p2p_mask, "dst_upi"]))
    )
    merchant_nodes = pd.Index(sorted(set(prepared.loc[~p2p_mask, "dst_upi"])))
    return {"user": user_nodes, "merchant": merchant_nodes}


def compute_cycle_completion_flags(transactions: pd.DataFrame) -> pd.Series:
    """
    Flag edges that close a directed cycle.

    The check is temporal: when edge u->v arrives, it is marked as a cycle
    completion if a directed path v=>u already exists in the historical graph.
    """

    prepared = prepare_transactions_for_graph(transactions)
    adjacency: dict[str, set[str]] = {}
    flags: list[int] = []

    for row in prepared.itertuples(index=False):
        closes_cycle = int(_has_path(row.dst_upi, row.src_upi, adjacency))
        flags.append(closes_cycle)
        adjacency.setdefault(row.src_upi, set()).add(row.dst_upi)

    return pd.Series(flags, index=prepared.index, dtype=int)


def edge_feature_matrix(transactions: pd.DataFrame) -> np.ndarray:
    """Return the ordered edge feature matrix as a numpy array."""

    featured = annotate_transactions_with_edge_features(transactions)
    return featured.loc[:, EDGE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)


def _build_node_feature_frame(
    node_ids: pd.Index,
    role: str,
    transactions: pd.DataFrame,
    centrality_scores: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> pd.DataFrame:
    if node_ids.empty:
        empty_frame = pd.DataFrame(columns=NODE_FEATURE_COLUMNS, index=node_ids)
        empty_frame.index.name = "upi_id"
        return empty_frame

    split_end = transactions["timestamp"].max()
    account_age = _compute_account_age_feature(
        node_ids=node_ids,
        role=role,
        transactions=transactions,
        split_end=split_end,
        config=config,
    )
    kyc_tier = _compute_kyc_tier_feature(
        node_ids=node_ids,
        role=role,
        transactions=transactions,
        config=config,
    )
    daily_avg_spend = _compute_daily_avg_spend_feature(
        node_ids=node_ids,
        role=role,
        transactions=transactions,
    )
    pagerank_local = _normalize_score_vector(
        np.array([centrality_scores["pagerank"].get(node_id, 0.0) for node_id in node_ids])
    )
    betweenness_central = _normalize_score_vector(
        np.array([centrality_scores["betweenness"].get(node_id, 0.0) for node_id in node_ids])
    )

    feature_frame = pd.DataFrame(
        {
            "account_age": account_age,
            "kyc_tier": kyc_tier,
            "daily_avg_spend": daily_avg_spend,
            "pagerank_local": pagerank_local,
            "betweenness_central": betweenness_central,
        },
        index=node_ids,
    )
    feature_frame.index.name = "upi_id"
    return feature_frame.astype(np.float32)


def _compute_account_age_feature(
    node_ids: pd.Index,
    role: str,
    transactions: pd.DataFrame,
    split_end: pd.Timestamp,
    config: dict[str, Any],
) -> np.ndarray:
    age_config = config["features"]["node"]["account_age_days"]
    min_days = int(age_config["min"])
    max_days = int(age_config["max"])
    appearance_map = _appearance_timestamps(transactions, role=role)

    ages = []
    for node_id in node_ids:
        first_seen = appearance_map.get(node_id, split_end)
        seeded_extra_days = min_days + _stable_int(
            node_id,
            "account_age",
            modulus=(max_days - min_days + 1),
        )
        creation_time = first_seen - pd.Timedelta(days=seeded_extra_days)
        age_days = max((split_end - creation_time).total_seconds() / 86400.0, float(min_days))
        ages.append(min(age_days / max_days, 1.0))

    return np.array(ages, dtype=np.float32)


def _compute_kyc_tier_feature(
    node_ids: pd.Index,
    role: str,
    transactions: pd.DataFrame,
    config: dict[str, Any],
) -> np.ndarray:
    distributions = config["features"]["node"]["kyc_distribution"]
    merchant_categories = (
        transactions.loc[transactions["merchant_type"].ne("peer_to_peer"), ["dst_upi", "merchant_type"]]
        .drop_duplicates("dst_upi")
        .set_index("dst_upi")["merchant_type"]
        .to_dict()
    )

    tiers = []
    for node_id in node_ids:
        distribution_key = "user"
        if role == "merchant":
            distribution_key = merchant_categories.get(node_id, "standard_retail")
        tiers.append(
            _sample_kyc_tier(
                distributions[distribution_key],
                key_parts=(role, node_id, distribution_key),
            )
        )
    return np.array(tiers, dtype=np.float32)


def _compute_daily_avg_spend_feature(
    node_ids: pd.Index,
    role: str,
    transactions: pd.DataFrame,
) -> np.ndarray:
    daily_volume = _daily_volume_by_node(transactions=transactions, role=role)
    split_start = transactions["timestamp"].min().floor("D")
    split_end = transactions["timestamp"].max().floor("D")
    full_days = pd.date_range(split_start, split_end, freq="D")

    averages = []
    for node_id in node_ids:
        node_series = daily_volume.get(node_id)
        if node_series is None:
            averages.append(0.0)
            continue
        aligned = node_series.reindex(full_days, fill_value=0.0)
        averages.append(float(aligned.rolling(window=30, min_periods=1).mean().iloc[-1]))

    log_scaled = np.log1p(np.array(averages, dtype=np.float32))
    return _normalize_score_vector(log_scaled)


def _daily_volume_by_node(
    transactions: pd.DataFrame,
    role: str,
) -> dict[str, pd.Series]:
    if role == "user":
        grouped = (
            transactions.assign(day=transactions["timestamp"].dt.floor("D"))
            .groupby(["src_upi", "day"])["amount_clipped"]
            .sum()
        )
    else:
        merchant_rows = transactions.loc[transactions["merchant_type"].ne("peer_to_peer")]
        grouped = (
            merchant_rows.assign(day=merchant_rows["timestamp"].dt.floor("D"))
            .groupby(["dst_upi", "day"])["amount_clipped"]
            .sum()
        )

    return {
        node_id: series.droplevel(0).sort_index()
        for node_id, series in grouped.groupby(level=0, sort=False)
    }


def _appearance_timestamps(
    transactions: pd.DataFrame,
    role: str,
) -> dict[str, pd.Timestamp]:
    if role == "user":
        src_seen = transactions.groupby("src_upi", sort=False)["timestamp"].min()
        dst_seen = (
            transactions.loc[transactions["merchant_type"].eq("peer_to_peer")]
            .groupby("dst_upi", sort=False)["timestamp"]
            .min()
        )
        combined = pd.concat(
            [src_seen.rename_axis("upi_id"), dst_seen.rename_axis("upi_id")]
        ).groupby(level=0).min()
    else:
        combined = (
            transactions.loc[transactions["merchant_type"].ne("peer_to_peer")]
            .groupby("dst_upi", sort=False)["timestamp"]
            .min()
        )
    return combined.to_dict()


def _compute_centrality_scores(
    graph: nx.DiGraph,
    config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    if graph.number_of_nodes() == 0:
        return {"pagerank": {}, "betweenness": {}}

    pagerank = _compute_pagerank_scores(graph, config=config)

    sample_size = int(config["graph"]["betweenness_sample_size"])
    if graph.number_of_nodes() > sample_size:
        betweenness = nx.betweenness_centrality(
            graph,
            k=sample_size,
            normalized=True,
            weight=None,
            seed=int(config["data"]["random_seed"]),
        )
    else:
        betweenness = nx.betweenness_centrality(
            graph,
            normalized=True,
            weight=None,
        )

    return {"pagerank": pagerank, "betweenness": betweenness}


def _compute_pagerank_scores(
    graph: nx.DiGraph,
    config: dict[str, Any],
) -> dict[str, float]:
    try:
        return nx.pagerank(
            graph,
            alpha=float(config["graph"]["pagerank_alpha"]),
            weight="weight",
            max_iter=int(config["graph"]["pagerank_max_iter"]),
            tol=float(config["graph"]["pagerank_tol"]),
        )
    except ModuleNotFoundError:
        pass

    alpha = float(config["graph"]["pagerank_alpha"])
    max_iter = int(config["graph"]["pagerank_max_iter"])
    tol = float(config["graph"]["pagerank_tol"])
    nodes = list(graph.nodes())
    node_index = {node_id: index for index, node_id in enumerate(nodes)}
    node_count = len(nodes)
    ranks = np.full(node_count, 1.0 / node_count, dtype=np.float64)
    out_weight = np.zeros(node_count, dtype=np.float64)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]

    for source, target, attributes in graph.edges(data=True):
        source_index = node_index[source]
        target_index = node_index[target]
        weight = float(attributes.get("weight", 1.0))
        adjacency[source_index].append((target_index, weight))
        out_weight[source_index] += weight

    teleport = (1.0 - alpha) / node_count
    dangling_mask = out_weight == 0.0

    for _ in range(max_iter):
        updated = np.full(node_count, teleport, dtype=np.float64)
        dangling_mass = alpha * ranks[dangling_mask].sum() / node_count
        updated += dangling_mass

        for source_index, neighbors in enumerate(adjacency):
            if not neighbors or out_weight[source_index] == 0.0:
                continue
            share = alpha * ranks[source_index] / out_weight[source_index]
            for target_index, weight in neighbors:
                updated[target_index] += share * weight

        if np.abs(updated - ranks).sum() < tol:
            ranks = updated
            break
        ranks = updated

    return {
        node_id: float(ranks[node_index[node_id]])
        for node_id in nodes
    }


def _build_interaction_graph(transactions: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in transactions.itertuples(index=False):
        existing_weight = graph[row.src_upi][row.dst_upi]["weight"] if graph.has_edge(row.src_upi, row.dst_upi) else 0.0
        graph.add_edge(row.src_upi, row.dst_upi, weight=existing_weight + float(row.amount_clipped))
    return graph


def _compute_source_delta_seconds(transactions: pd.DataFrame) -> pd.Series:
    delta = (
        transactions.groupby("src_upi", sort=False)["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
        .clip(lower=0.0)
    )
    return delta.astype(np.float32)


def _normalize_score_vector(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    max_value = float(values.max())
    if max_value <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return (values / max_value).astype(np.float32)


def _sample_kyc_tier(
    distribution: dict[str, float],
    key_parts: tuple[str, str, str],
) -> int:
    threshold = _stable_fraction(*key_parts)
    cumulative = 0.0
    chosen_tier = 2
    for tier_key, probability in distribution.items():
        cumulative += float(probability)
        chosen_tier = int(tier_key.split("_")[-1])
        if threshold <= cumulative:
            break
    return chosen_tier


def _has_path(
    source: str,
    target: str,
    adjacency: dict[str, set[str]],
) -> bool:
    if source == target:
        return True

    stack = [source]
    seen = {source}
    while stack:
        node = stack.pop()
        for neighbor in adjacency.get(node, set()):
            if neighbor == target:
                return True
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return False


def _stable_int(*parts: Any, modulus: int) -> int:
    return int(_stable_fraction(*parts) * modulus) % modulus


def _stable_fraction(*parts: Any) -> float:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") / float(2**64)
