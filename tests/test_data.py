from pathlib import Path

import pandas as pd
import torch

from src.data.feature_engineer import (
    compute_cycle_completion_flags,
    edge_feature_matrix,
)
from src.data.graph_builder import build_graph_splits, split_transactions_chronologically
from src.data.partitioner import partition_training_graph
from src.data.paysim_loader import load_config, load_paysim_as_upi
from src.data.smote import apply_graph_smote


def _write_synthetic_paysim_csv(tmp_path: Path) -> Path:
    total_rows = 2000
    records = []
    transaction_types = ["PAYMENT", "DEBIT", "TRANSFER", "CASH_OUT", "CASH_IN"]

    for index in range(total_rows):
        txn_type = transaction_types[index % len(transaction_types)]
        amount = 600.0 + float((index % 37) * 275.0)
        name_orig = f"C{index % 450:06d}"
        name_dest = f"{'C' if txn_type == 'TRANSFER' else 'M'}{(index * 7) % 320:06d}"
        is_fraud = 0
        is_flagged_fraud = 0

        if index == 10:
            txn_type = "TRANSFER"
            amount = 250000.0
            name_orig = "C999999"
            name_dest = "C888888"
            is_fraud = 1
            is_flagged_fraud = 1
        elif index == 777:
            txn_type = "PAYMENT"
            amount = 17500.0
            name_orig = "C777777"
            name_dest = "M777777"
            is_fraud = 1
        elif index == 1245:
            txn_type = "TRANSFER"
            amount = 150000.0
            name_orig = "C124500"
            name_dest = "C124599"

        records.append(
            {
                "step": index % 720,
                "type": txn_type,
                "amount": amount,
                "nameOrig": name_orig,
                "nameDest": name_dest,
                "isFraud": is_fraud,
                "isFlaggedFraud": is_flagged_fraud,
            }
        )

    frame = pd.DataFrame.from_records(records)
    csv_path = tmp_path / "paysim_sample.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def test_loader_enforces_fraud_rate_amount_cap_and_time_span(tmp_path: Path) -> None:
    csv_path = _write_synthetic_paysim_csv(tmp_path)

    adapted = load_paysim_as_upi(csv_path)

    fraud_rate = adapted["is_fraud"].mean()
    assert 0.001 <= fraud_rate <= 0.002
    assert adapted["amount_clipped"].max() <= 100000.0
    assert adapted["timestamp"].max() - adapted["timestamp"].min() == pd.Timedelta(days=30)
    assert list(adapted.columns) == [
        "txn_id",
        "src_upi",
        "dst_upi",
        "amount_clipped",
        "timestamp",
        "merchant_type",
        "mcc_weight",
        "is_fraud",
    ]


def test_large_p2p_transfer_is_split_across_multiple_vpas(tmp_path: Path) -> None:
    csv_path = _write_synthetic_paysim_csv(tmp_path)

    adapted = load_paysim_as_upi(csv_path)
    split_rows = adapted.loc[adapted["txn_id"].str.startswith("TXN_0000010_")]

    assert len(split_rows) == 3
    assert split_rows["dst_upi"].nunique() == 3
    assert split_rows["merchant_type"].eq("peer_to_peer").all()
    assert sorted(split_rows["amount_clipped"].tolist()) == [50000.0, 100000.0, 100000.0]


def test_mcc_weights_match_configured_mapping(tmp_path: Path) -> None:
    csv_path = _write_synthetic_paysim_csv(tmp_path)

    adapted = load_paysim_as_upi(csv_path)
    config = load_config()
    expected_weights = config["data"]["merchant_risk_weights"]

    observed = (
        adapted.groupby("merchant_type", sort=False)["mcc_weight"]
        .unique()
        .apply(list)
        .to_dict()
    )

    for merchant_type, weights in observed.items():
        assert weights == [expected_weights[merchant_type]]


def test_edge_feature_matrix_has_expected_dimension(tmp_path: Path) -> None:
    csv_path = _write_synthetic_paysim_csv(tmp_path)

    adapted = load_paysim_as_upi(csv_path)
    features = edge_feature_matrix(adapted)

    assert features.shape[0] == len(adapted)
    assert features.shape[1] == 5


def test_cycle_flag_identifies_a_to_b_to_c_to_a_pattern() -> None:
    frame = pd.DataFrame(
        [
            {
                "txn_id": "TXN_1",
                "src_upi": "ua@okaxis",
                "dst_upi": "ub@oksbi",
                "amount_clipped": 10.0,
                "timestamp": "2024-01-01T00:00:00",
                "merchant_type": "peer_to_peer",
                "mcc_weight": 1.0,
                "is_fraud": 0,
            },
            {
                "txn_id": "TXN_2",
                "src_upi": "ub@oksbi",
                "dst_upi": "uc@okhdfcbank",
                "amount_clipped": 15.0,
                "timestamp": "2024-01-01T00:05:00",
                "merchant_type": "peer_to_peer",
                "mcc_weight": 1.0,
                "is_fraud": 0,
            },
            {
                "txn_id": "TXN_3",
                "src_upi": "uc@okhdfcbank",
                "dst_upi": "ua@okaxis",
                "amount_clipped": 20.0,
                "timestamp": "2024-01-01T00:10:00",
                "merchant_type": "peer_to_peer",
                "mcc_weight": 1.0,
                "is_fraud": 1,
            },
        ]
    )

    cycle_flags = compute_cycle_completion_flags(frame)

    assert cycle_flags.tolist() == [0, 0, 1]


def test_graph_builder_creates_chronological_splits_and_five_edge_features(
    tmp_path: Path,
) -> None:
    csv_path = _write_synthetic_paysim_csv(tmp_path)
    adapted = load_paysim_as_upi(csv_path)
    config = load_config()
    output_dir = tmp_path / "processed"

    splits = split_transactions_chronologically(adapted)
    assert splits["train"]["timestamp"].max() < splits["val"]["timestamp"].min()
    assert splits["val"]["timestamp"].max() < splits["test"]["timestamp"].min()

    graphs = build_graph_splits(adapted, output_dir=output_dir, persist=True)

    for split_name, graph in graphs.items():
        expected_path = output_dir / config["graph"]["processed_files"][split_name]
        assert expected_path.is_file()

        assert graph["user"].x.shape[1] == 5
        if graph["merchant"].x.numel():
            assert graph["merchant"].x.shape[1] == 5

        edge_feature_dims = {graph[edge_type].edge_attr.shape[1] for edge_type in graph.edge_types}
        assert edge_feature_dims == {5}


def test_graph_smote_lifts_training_fraud_rate_into_target_band(tmp_path: Path) -> None:
    csv_path = _write_synthetic_paysim_csv(tmp_path)
    adapted = load_paysim_as_upi(csv_path)
    train_graph = build_graph_splits(adapted, output_dir=tmp_path / "graphs", persist=False)["train"]
    output_path = tmp_path / "graph_train_smote.pt"

    augmented = apply_graph_smote(train_graph, output_path=output_path, persist=True)
    labels = torch.cat(
        [augmented[edge_type].y.float() for edge_type in augmented.edge_types if augmented[edge_type].y.numel()],
        dim=0,
    )
    fraud_rate = float(labels.mean().item())

    assert 0.01 <= fraud_rate <= 0.05
    assert output_path.is_file()
    assert augmented["user"].num_nodes > train_graph["user"].num_nodes
    assert int(augmented["user"].synthetic_mask.sum().item()) > 0

    original_bounds_min = train_graph["user"].x.min(dim=0).values
    original_bounds_max = train_graph["user"].x.max(dim=0).values
    synthetic_features = augmented["user"].x[augmented["user"].synthetic_mask]

    assert torch.all(synthetic_features >= (original_bounds_min - 1e-6))
    assert torch.all(synthetic_features <= (original_bounds_max + 1e-6))


def test_partitioner_creates_fifty_cluster_assignments(tmp_path: Path) -> None:
    csv_path = _write_synthetic_paysim_csv(tmp_path)
    adapted = load_paysim_as_upi(csv_path)
    train_graph = build_graph_splits(adapted, output_dir=tmp_path / "graphs", persist=False)["train"]
    augmented = apply_graph_smote(train_graph, output_path=tmp_path / "graph_train_smote.pt", persist=False)
    assignment_path = tmp_path / "cluster_assignments.pkl"

    assignments = partition_training_graph(augmented, output_path=assignment_path, persist=True)
    combined = torch.cat([assignments["user"], assignments["merchant"]], dim=0)

    assert assignment_path.is_file()
    assert combined.unique().numel() == 50
    assert combined.shape[0] == augmented["user"].num_nodes + augmented["merchant"].num_nodes
    assert assignments["method"] in {"pymetis", "balanced_neighborhood_fallback"}
