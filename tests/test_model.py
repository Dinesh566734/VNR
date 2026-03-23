import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from src.data.paysim_loader import load_config
from src.data.graph_builder import build_hetero_graph
from src.models.explainer import SentinelExplainer
from src.models.focal_loss import FocalLoss, binary_focal_loss
from src.models.gat import SentinelGAT
from src.training.ablation import run_ablation_studies, run_baseline_benchmarks
from src.training.eval import evaluate_test_graph
from src.training.train import train_model


@pytest.fixture()
def sample_graph() -> object:
    transactions = pd.DataFrame(
        [
            {
                "txn_id": "TXN_001",
                "src_upi": "ua@okaxis",
                "dst_upi": "ub@oksbi",
                "amount_clipped": 1200.0,
                "timestamp": "2024-01-01T09:00:00",
                "merchant_type": "peer_to_peer",
                "mcc_weight": 1.0,
                "is_fraud": 0,
            },
            {
                "txn_id": "TXN_002",
                "src_upi": "uc@ybl",
                "dst_upi": "ub@oksbi",
                "amount_clipped": 1800.0,
                "timestamp": "2024-01-01T09:05:00",
                "merchant_type": "peer_to_peer",
                "mcc_weight": 1.0,
                "is_fraud": 1,
            },
            {
                "txn_id": "TXN_003",
                "src_upi": "ud@okicici",
                "dst_upi": "ub@oksbi",
                "amount_clipped": 950.0,
                "timestamp": "2024-01-01T09:10:00",
                "merchant_type": "peer_to_peer",
                "mcc_weight": 1.0,
                "is_fraud": 0,
            },
            {
                "txn_id": "TXN_004",
                "src_upi": "ub@oksbi",
                "dst_upi": "ua@okaxis",
                "amount_clipped": 400.0,
                "timestamp": "2024-01-01T09:20:00",
                "merchant_type": "peer_to_peer",
                "mcc_weight": 1.0,
                "is_fraud": 0,
            },
            {
                "txn_id": "TXN_005",
                "src_upi": "ua@okaxis",
                "dst_upi": "mshop@paytm",
                "amount_clipped": 2500.0,
                "timestamp": "2024-01-01T10:00:00",
                "merchant_type": "standard_retail",
                "mcc_weight": 1.0,
                "is_fraud": 0,
            },
            {
                "txn_id": "TXN_006",
                "src_upi": "ub@oksbi",
                "dst_upi": "mshop@paytm",
                "amount_clipped": 4100.0,
                "timestamp": "2024-01-01T10:10:00",
                "merchant_type": "standard_retail",
                "mcc_weight": 1.0,
                "is_fraud": 1,
            },
            {
                "txn_id": "TXN_007",
                "src_upi": "uc@ybl",
                "dst_upi": "mgame@ibl",
                "amount_clipped": 900.0,
                "timestamp": "2024-01-01T10:15:00",
                "merchant_type": "gaming_wallet",
                "mcc_weight": 1.5,
                "is_fraud": 1,
            },
            {
                "txn_id": "TXN_008",
                "src_upi": "ud@okicici",
                "dst_upi": "mcrypto@ibl",
                "amount_clipped": 5000.0,
                "timestamp": "2024-01-01T10:25:00",
                "merchant_type": "crypto_offshore",
                "mcc_weight": 2.0,
                "is_fraud": 1,
            },
        ]
    )
    return build_hetero_graph(transactions)


def test_focal_loss_is_positive_and_finite_for_probabilities() -> None:
    predictions = torch.tensor([0.92, 0.18, 0.73, 0.04], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    loss = FocalLoss()(predictions, targets)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_focal_loss_is_positive_and_finite_for_logits() -> None:
    logits = torch.tensor([2.5, -1.8, 0.4, -3.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    loss = FocalLoss(from_logits=True)(logits, targets)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_focal_loss_down_weights_easy_examples_relative_to_hard_examples() -> None:
    targets = torch.tensor([1.0, 0.0], dtype=torch.float32)
    easy_predictions = torch.tensor([0.99, 0.01], dtype=torch.float32)
    hard_predictions = torch.tensor([0.55, 0.45], dtype=torch.float32)

    easy_loss = binary_focal_loss(easy_predictions, targets, reduction="mean")
    hard_loss = binary_focal_loss(hard_predictions, targets, reduction="mean")

    assert hard_loss.item() > easy_loss.item()


def test_focal_loss_matches_bce_when_gamma_is_zero() -> None:
    predictions = torch.tensor([0.82, 0.27, 0.61], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    focal = binary_focal_loss(predictions, targets, alpha=0.5, gamma=0.0, reduction="mean")
    bce = torch.nn.functional.binary_cross_entropy(predictions, targets)

    assert math.isclose(focal.item(), 0.5 * bce.item(), rel_tol=1e-6, abs_tol=1e-6)


def test_focal_loss_rejects_invalid_hyperparameters() -> None:
    predictions = torch.tensor([0.8], dtype=torch.float32)
    targets = torch.tensor([1.0], dtype=torch.float32)

    with pytest.raises(ValueError):
        binary_focal_loss(predictions, targets, alpha=1.5)

    with pytest.raises(ValueError):
        binary_focal_loss(predictions, targets, gamma=-1.0)


def test_gat_outputs_probability_per_edge(sample_graph: object) -> None:
    model = SentinelGAT.from_graph(sample_graph)
    model.eval()

    predictions = model.predict_all_edges(sample_graph)
    total_edges = sum(sample_graph[edge_type].edge_index.shape[1] for edge_type in sample_graph.edge_types)

    assert predictions.shape == (total_edges, 1)
    assert torch.all(predictions >= 0.0)
    assert torch.all(predictions <= 1.0)


def test_gat_attention_weights_sum_to_one_per_destination(sample_graph: object) -> None:
    model = SentinelGAT.from_graph(sample_graph)
    model.eval()

    _, attention_weights = model(sample_graph, return_attention_weights=True)

    for edge_type, alpha in attention_weights.items():
        edge_index = sample_graph[edge_type].edge_index
        destinations = edge_index[1]
        for head in range(alpha.shape[1]):
            for destination in destinations.unique().tolist():
                mask = destinations == destination
                weight_sum = alpha[mask, head].sum().item()
                assert math.isclose(weight_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5)


def test_train_model_runs_and_writes_checkpoint(sample_graph: object, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "sentinel_gat_best.pt"
    cluster_assignments = {
        "user": torch.zeros(sample_graph["user"].num_nodes, dtype=torch.long),
        "merchant": torch.zeros(sample_graph["merchant"].num_nodes, dtype=torch.long),
        "method": "manual",
        "num_clusters": 1,
    }

    artifacts = train_model(
        train_graph=sample_graph,
        val_graph=sample_graph,
        cluster_assignments=cluster_assignments,
        checkpoint_path=checkpoint_path,
        device="cpu",
        max_epochs=3,
        patience=2,
        verbose=False,
    )

    assert checkpoint_path.is_file()
    assert artifacts.best_epoch >= 1
    assert artifacts.best_val_f1 >= 0.0
    assert artifacts.history
    assert all(torch.isfinite(torch.tensor(metric.train_loss)) for metric in artifacts.history)


def test_evaluate_test_graph_returns_metrics(sample_graph: object) -> None:
    model = SentinelGAT.from_graph(sample_graph)
    model.eval()

    report = evaluate_test_graph(
        model,
        sample_graph,
        device="cpu",
        latency_trials=2,
        warmup_trials=1,
    )

    assert 0.0 <= report.precision <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert 0.0 <= report.f1_score <= 1.0
    assert 0.0 <= report.auc_roc <= 1.0
    assert report.average_latency_ms >= 0.0
    assert report.monthly_loss_inr >= 0.0


def test_ablation_runner_returns_markdown(sample_graph: object) -> None:
    cluster_assignments = {
        "user": torch.zeros(sample_graph["user"].num_nodes, dtype=torch.long),
        "merchant": torch.zeros(sample_graph["merchant"].num_nodes, dtype=torch.long),
        "method": "manual",
        "num_clusters": 1,
    }

    def stub_train_fn(**kwargs: object) -> object:
        model_kwargs = kwargs.get("model_kwargs") or {}
        model = SentinelGAT.from_graph(sample_graph, **model_kwargs)
        return type(
            "Artifacts",
            (),
            {
                "model": model,
            },
        )()

    report = run_ablation_studies(
        train_graph=sample_graph,
        val_graph=sample_graph,
        cluster_assignments=cluster_assignments,
        device="cpu",
        max_epochs=1,
        patience=1,
        verbose=False,
        attention_heads=(1, 2),
        depths=(1, 2),
        latency_trials=1,
        train_fn=stub_train_fn,
    )

    assert len(report.results) == 6
    assert "| Experiment | Variant | Precision | Recall | F1 | AUC-ROC | FPR | Monthly Loss (INR) | Latency (ms) |" in report.markdown_table


def test_explainer_returns_required_keys(sample_graph: object) -> None:
    model = SentinelGAT.from_graph(sample_graph)
    model.eval()
    relation = sample_graph.edge_types[0]
    explainer = SentinelExplainer(model, gnn_explainer_epochs=2)

    explanation = explainer.explain_transaction(
        sample_graph,
        edge_type=relation,
        edge_index=0,
    )

    assert {
        "txn_id",
        "risk_score",
        "decision",
        "top_features",
        "critical_edges",
        "fraud_pattern",
        "analyst_summary",
    }.issubset(explanation.keys())
    assert explanation["decision"] in {"ALLOW", "BLOCK"}
    assert isinstance(explanation["top_features"], list)
    assert isinstance(explanation["critical_edges"], list)


def test_baseline_benchmarks_write_markdown_report(sample_graph: object, tmp_path: Path) -> None:
    config = load_config()
    config["benchmark"]["logistic_regression"]["max_iter"] = 80
    config["benchmark"]["random_forest"]["n_estimators"] = 24
    config["benchmark"]["random_forest"]["max_depth"] = 4
    config["benchmark"]["xgboost"]["n_estimators"] = 16
    config["benchmark"]["xgboost"]["max_depth"] = 3
    config["benchmark"]["graphsage"]["hidden_channels"] = 16
    config["benchmark"]["graphsage"]["max_epochs"] = 3
    config["benchmark"]["graphsage"]["early_stopping_patience"] = 2
    config_path = tmp_path / "benchmark_hparams.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_path = tmp_path / "benchmark.md"

    report = run_baseline_benchmarks(
        train_graph=sample_graph,
        val_graph=sample_graph,
        test_graph=sample_graph,
        config_path=str(config_path),
        output_path=output_path,
        device="cpu",
        max_sentinel_epochs=2,
        patience=1,
        verbose=False,
    )

    assert output_path.is_file()
    assert len(report.results) == 5
    assert "| Model | Type | Precision | Recall | F1 | AUC-ROC | Latency (ms) |" in report.markdown_table
    assert "Sentinel-UPI" in report.markdown_table
