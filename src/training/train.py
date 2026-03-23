"""Training loop for Sentinel-UPI."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch_geometric.data import HeteroData

from src.data.graph_builder import build_graph_splits
from src.data.partitioner import partition_training_graph
from src.data.paysim_loader import PROJECT_ROOT, load_config, load_paysim_as_upi
from src.data.smote import apply_graph_smote
from src.models.focal_loss import FocalLoss
from src.models.gat import SentinelGAT


EdgeType = tuple[str, str, str]


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_f1: float
    val_precision: float
    val_recall: float


@dataclass
class TrainingArtifacts:
    model: SentinelGAT
    history: list[EpochMetrics]
    best_epoch: int
    best_val_f1: float
    checkpoint_path: Path | None
    device: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint_path"] = str(self.checkpoint_path) if self.checkpoint_path else None
        return payload


@dataclass
class RawPipelineArtifacts:
    adapted_rows: int
    split_edge_counts: dict[str, int]
    checkpoint_path: Path | None
    evaluation_path: Path
    training_summary_path: Path
    best_epoch: int
    best_val_f1: float
    test_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapted_rows": self.adapted_rows,
            "split_edge_counts": self.split_edge_counts,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "evaluation_path": str(self.evaluation_path),
            "training_summary_path": str(self.training_summary_path),
            "best_epoch": self.best_epoch,
            "best_val_f1": self.best_val_f1,
            "test_report": self.test_report,
        }


def train_model(
    train_graph: HeteroData,
    val_graph: HeteroData,
    *,
    cluster_assignments: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    criterion: nn.Module | None = None,
    save_checkpoint: bool = True,
    verbose: bool = True,
) -> TrainingArtifacts:
    """Train the edge-aware GAT on partitioned graph batches."""

    config = load_config(config_path)
    training_config = config["training"]
    loss_config = config["model"]["focal_loss"]
    resolved_device = _resolve_device(device)

    model = SentinelGAT.from_graph(
        train_graph,
        config_path=config_path,
        **(model_kwargs or {}),
    ).to(resolved_device)
    criterion = criterion or FocalLoss(
        alpha=float(loss_config["alpha"]),
        gamma=float(loss_config["gamma"]),
        from_logits=True,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    if cluster_assignments is None:
        cluster_assignments = partition_training_graph(train_graph, config_path=config_path, persist=False)

    resolved_checkpoint = (
        _resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
            config=config,
        )
        if save_checkpoint
        else None
    )
    total_epochs = int(max_epochs if max_epochs is not None else training_config["max_epochs"])
    early_stopping_patience = int(
        patience if patience is not None else training_config["early_stopping_patience"]
    )
    threshold = float(training_config["decision_threshold"])

    history: list[EpochMetrics] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_val_f1 = float("-inf")
    patience_counter = 0

    for epoch in range(1, total_epochs + 1):
        model.train()
        batch_losses: list[float] = []

        for cluster_graph in iter_cluster_subgraphs(train_graph, cluster_assignments):
            if total_edge_count(cluster_graph) == 0:
                continue

            batch = cluster_graph.to(resolved_device)
            optimizer.zero_grad()
            logits_by_edge = model(batch, return_logits=True)
            logits, labels = flatten_edge_predictions(logits_by_edge, batch)
            if labels.numel() == 0:
                continue

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(sum(batch_losses) / len(batch_losses)) if batch_losses else 0.0
        validation = evaluate_model(
            model=model,
            graph=val_graph,
            criterion=criterion,
            device=resolved_device,
            threshold=threshold,
        )
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=validation["loss"],
            val_f1=validation["f1"],
            val_precision=validation["precision"],
            val_recall=validation["recall"],
        )
        history.append(epoch_metrics)

        if verbose:
            print(format_epoch_log(epoch_metrics))

        if epoch_metrics.val_f1 > best_val_f1:
            best_val_f1 = epoch_metrics.val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            if resolved_checkpoint is not None:
                _save_checkpoint(
                    checkpoint_path=resolved_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch_metrics=epoch_metrics,
                    config=config,
                )
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingArtifacts(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_f1=float(best_val_f1 if best_val_f1 != float("-inf") else 0.0),
        checkpoint_path=resolved_checkpoint,
        device=str(resolved_device),
    )


def evaluate_model(
    *,
    model: SentinelGAT,
    graph: HeteroData,
    criterion: FocalLoss,
    device: str | torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate loss and binary metrics on a full graph split."""

    model.eval()
    with torch.no_grad():
        graph_on_device = graph.to(device)
        logits_by_edge = model(graph_on_device, return_logits=True)
        logits, labels = flatten_edge_predictions(logits_by_edge, graph_on_device)
        if labels.numel() == 0:
            return {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).float()
        metrics = binary_classification_metrics(predictions=predictions, labels=labels)
        metrics["loss"] = float(loss.item())
        return metrics


def iter_cluster_subgraphs(
    graph: HeteroData,
    cluster_assignments: dict[str, Any],
) -> Iterator[HeteroData]:
    """Yield cluster-local subgraphs used as mini-batches."""

    cluster_ids = sorted(
        {
            int(cluster_id)
            for key in ("user", "merchant")
            if key in cluster_assignments
            for cluster_id in cluster_assignments[key].tolist()
            if int(cluster_id) >= 0
        }
    )

    for cluster_id in cluster_ids:
        user_nodes = torch.nonzero(cluster_assignments["user"] == cluster_id, as_tuple=False).view(-1)
        merchant_nodes = torch.nonzero(
            cluster_assignments["merchant"] == cluster_id, as_tuple=False
        ).view(-1)
        cluster_graph = extract_cluster_subgraph(
            graph=graph,
            user_nodes=user_nodes,
            merchant_nodes=merchant_nodes,
        )
        if total_edge_count(cluster_graph) > 0:
            yield cluster_graph


def extract_cluster_subgraph(
    *,
    graph: HeteroData,
    user_nodes: Tensor,
    merchant_nodes: Tensor,
) -> HeteroData:
    """Create a cluster-local HeteroData subgraph with remapped node indices."""

    subgraph = HeteroData()
    user_nodes = user_nodes.long()
    merchant_nodes = merchant_nodes.long()

    subgraph["user"].x = graph["user"].x[user_nodes].clone()
    subgraph["user"].num_nodes = int(user_nodes.numel())
    subgraph["user"].upi_id = [graph["user"].upi_id[index] for index in user_nodes.tolist()]
    if hasattr(graph["user"], "synthetic_mask"):
        subgraph["user"].synthetic_mask = graph["user"].synthetic_mask[user_nodes].clone()

    subgraph["merchant"].x = graph["merchant"].x[merchant_nodes].clone()
    subgraph["merchant"].num_nodes = int(merchant_nodes.numel())
    subgraph["merchant"].upi_id = [
        graph["merchant"].upi_id[index] for index in merchant_nodes.tolist()
    ]

    user_map = _build_index_map(graph["user"].num_nodes, user_nodes)
    merchant_map = _build_index_map(graph["merchant"].num_nodes, merchant_nodes)

    for edge_type in graph.edge_types:
        edge_store = graph[edge_type]
        dst_map = user_map if edge_type[2] == "user" else merchant_map
        src_local = user_map[edge_store.edge_index[0]]
        dst_local = dst_map[edge_store.edge_index[1]]
        edge_mask = (src_local >= 0) & (dst_local >= 0)

        if not edge_mask.any():
            subgraph[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            subgraph[edge_type].edge_attr = torch.empty(
                (0, edge_store.edge_attr.shape[1]),
                dtype=edge_store.edge_attr.dtype,
            )
            subgraph[edge_type].y = torch.empty((0,), dtype=edge_store.y.dtype)
            subgraph[edge_type].timestamp = torch.empty((0,), dtype=edge_store.timestamp.dtype)
            subgraph[edge_type].txn_id = []
            continue

        kept_indices = torch.nonzero(edge_mask, as_tuple=False).view(-1)
        subgraph[edge_type].edge_index = torch.stack(
            [src_local[edge_mask], dst_local[edge_mask]],
            dim=0,
        )
        subgraph[edge_type].edge_attr = edge_store.edge_attr[edge_mask].clone()
        subgraph[edge_type].y = edge_store.y[edge_mask].clone()
        subgraph[edge_type].timestamp = edge_store.timestamp[edge_mask].clone()
        subgraph[edge_type].txn_id = [
            edge_store.txn_id[index]
            for index in kept_indices.tolist()
        ]

    subgraph.validate(raise_on_error=True)
    return subgraph


def flatten_edge_predictions(
    predictions_by_edge: dict[EdgeType, Tensor],
    graph: HeteroData,
) -> tuple[Tensor, Tensor]:
    """Flatten relation-wise edge outputs and labels into aligned tensors."""

    logits: list[Tensor] = []
    labels: list[Tensor] = []
    for edge_type in graph.edge_types:
        if edge_type not in predictions_by_edge:
            continue
        edge_predictions = predictions_by_edge[edge_type].reshape(-1)
        edge_labels = graph[edge_type].y.float().reshape(-1)
        if edge_labels.numel() == 0:
            continue
        logits.append(edge_predictions)
        labels.append(edge_labels)

    if not logits:
        empty = torch.empty((0,), dtype=torch.float32)
        return empty, empty
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def binary_classification_metrics(
    *,
    predictions: Tensor,
    labels: Tensor,
) -> dict[str, float]:
    """Compute precision, recall, and F1 for binary predictions."""

    predictions = predictions.float().reshape(-1)
    labels = labels.float().reshape(-1)
    true_positive = float(((predictions == 1.0) & (labels == 1.0)).sum().item())
    false_positive = float(((predictions == 1.0) & (labels == 0.0)).sum().item())
    false_negative = float(((predictions == 0.0) & (labels == 1.0)).sum().item())

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def total_edge_count(graph: HeteroData) -> int:
    """Return the total number of edges across all relations."""

    return int(sum(graph[edge_type].edge_index.shape[1] for edge_type in graph.edge_types))


def format_epoch_log(metrics: EpochMetrics) -> str:
    """Format per-epoch console logging to the requested shape."""

    return (
        f"Epoch {metrics.epoch:03d} | "
        f"Loss: {metrics.train_loss:.4f} | "
        f"Val F1: {metrics.val_f1:.3f} | "
        f"Prec: {metrics.val_precision:.3f} | "
        f"Rec: {metrics.val_recall:.3f}"
    )


def _build_index_map(size: int, kept_nodes: Tensor) -> Tensor:
    index_map = torch.full((int(size),), -1, dtype=torch.long)
    if kept_nodes.numel():
        index_map[kept_nodes] = torch.arange(kept_nodes.numel(), dtype=torch.long)
    return index_map


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_checkpoint_path(
    *,
    checkpoint_path: str | Path | None,
    config: dict[str, Any],
) -> Path:
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path)
    else:
        resolved = (
            PROJECT_ROOT
            / config["data"]["processed_dir"]
            / config["training"]["checkpoint_name"]
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    model: SentinelGAT,
    optimizer: AdamW,
    epoch_metrics: EpochMetrics,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch_metrics": asdict(epoch_metrics),
            "config": config,
        },
        checkpoint_path,
    )


def train_from_raw_paysim(
    csv_path: str | Path,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_rows: int | None = None,
    device: str | torch.device | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    verbose: bool = True,
) -> RawPipelineArtifacts:
    """Run the end-to-end training pipeline from a raw PaySim CSV."""

    from .eval import evaluate_test_graph, write_evaluation_report

    config = load_config(config_path)
    resolved_output_dir = _resolve_run_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    adapted = load_paysim_as_upi(csv_path, config_path=config_path, max_rows=max_rows)
    graphs = build_graph_splits(
        adapted,
        config_path=config_path,
        output_dir=resolved_output_dir,
        persist=True,
    )
    augmented_train = apply_graph_smote(
        graphs["train"],
        config_path=config_path,
        output_path=resolved_output_dir / "graph_train_smote.pt",
        persist=True,
    )
    cluster_assignments = partition_training_graph(
        augmented_train,
        config_path=config_path,
        output_path=resolved_output_dir / config["partitioner"]["assignment_file"],
        persist=True,
    )
    training_artifacts = train_model(
        train_graph=augmented_train,
        val_graph=graphs["val"],
        cluster_assignments=cluster_assignments,
        config_path=config_path,
        checkpoint_path=resolved_output_dir / config["training"]["checkpoint_name"],
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        verbose=verbose,
    )
    test_report = evaluate_test_graph(
        model=training_artifacts.model,
        graph=graphs["test"],
        config_path=config_path,
        device=device,
    )

    evaluation_path = write_evaluation_report(
        test_report,
        resolved_output_dir / "test_evaluation.json",
    )
    training_summary_path = resolved_output_dir / "training_summary.json"
    split_edge_counts = {
        split_name: total_edge_count(graph)
        for split_name, graph in graphs.items()
    }
    summary_payload = {
        "input_csv": str(Path(csv_path).resolve()),
        "max_rows": max_rows,
        "adapted_rows": int(len(adapted)),
        "split_edge_counts": split_edge_counts,
        "best_epoch": training_artifacts.best_epoch,
        "best_val_f1": training_artifacts.best_val_f1,
        "checkpoint_path": str(training_artifacts.checkpoint_path) if training_artifacts.checkpoint_path else None,
        "evaluation_report": test_report.to_dict(),
    }
    with training_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    return RawPipelineArtifacts(
        adapted_rows=int(len(adapted)),
        split_edge_counts=split_edge_counts,
        checkpoint_path=training_artifacts.checkpoint_path,
        evaluation_path=evaluation_path,
        training_summary_path=training_summary_path,
        best_epoch=training_artifacts.best_epoch,
        best_val_f1=training_artifacts.best_val_f1,
        test_report=test_report.to_dict(),
    )


def _resolve_run_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return PROJECT_ROOT / "data" / "processed" / "real_run"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Sentinel-UPI from a raw PaySim CSV.")
    parser.add_argument(
        "--csv-path",
        default=str(PROJECT_ROOT / "data" / "raw" / "paysim.csv"),
        help="Path to the raw PaySim CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "real_run"),
        help="Directory for graphs, checkpoint, and reports.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for the raw CSV to keep real-data runs tractable.",
    )
    parser.add_argument("--device", default=None, help="Training device, for example cpu or cuda.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-epoch logging.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    artifacts = train_from_raw_paysim(
        args.csv_path,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        device=args.device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        verbose=not args.quiet,
    )
    print(json.dumps(artifacts.to_dict(), indent=2))


if __name__ == "__main__":
    main()
