"""Evaluation utilities for Sentinel-UPI."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.data.paysim_loader import load_config
from src.models.focal_loss import FocalLoss
from src.models.gat import SentinelGAT
from .train import flatten_edge_predictions


@dataclass
class ConfusionMatrix:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


@dataclass
class EvaluationReport:
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    average_latency_ms: float
    confusion_matrix: ConfusionMatrix
    false_positive_rate: float
    monthly_loss_inr: float
    loss: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confusion_matrix"] = asdict(self.confusion_matrix)
        return payload


def evaluate_test_graph(
    model: SentinelGAT,
    graph: HeteroData,
    *,
    config_path: str | None = None,
    device: str | torch.device | None = None,
    threshold: float | None = None,
    latency_trials: int = 20,
    warmup_trials: int = 3,
) -> EvaluationReport:
    """Evaluate a trained model on a held-out graph split."""

    config = load_config(config_path)
    training_config = config["training"]
    loss_config = config["model"]["focal_loss"]
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    resolved_threshold = float(
        threshold if threshold is not None else training_config["decision_threshold"]
    )

    criterion = FocalLoss(
        alpha=float(loss_config["alpha"]),
        gamma=float(loss_config["gamma"]),
        from_logits=True,
    )
    model = model.to(resolved_device)
    model.eval()
    graph_on_device = graph.to(resolved_device)

    with torch.no_grad():
        logits_by_edge = model(graph_on_device, return_logits=True)
        logits, labels = flatten_edge_predictions(logits_by_edge, graph_on_device)
        if labels.numel() == 0:
            raise ValueError("Evaluation graph must contain at least one labeled edge.")

        loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= resolved_threshold).float()

    confusion = confusion_from_predictions(predictions=predictions, labels=labels)
    precision = precision_from_confusion(confusion)
    recall = recall_from_confusion(confusion)
    f1_score = f1_from_precision_recall(precision, recall)
    false_positive_rate = false_positive_rate_from_confusion(confusion)
    auc_roc = binary_roc_auc(probabilities=probabilities, labels=labels)
    average_latency_ms = average_inference_latency_ms(
        model=model,
        graph=graph,
        device=resolved_device,
        trials=latency_trials,
        warmup_trials=warmup_trials,
    )
    monthly_loss_inr = float(confusion.false_positive * 200 + confusion.false_negative * 15000)

    return EvaluationReport(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        auc_roc=auc_roc,
        average_latency_ms=average_latency_ms,
        confusion_matrix=confusion,
        false_positive_rate=false_positive_rate,
        monthly_loss_inr=monthly_loss_inr,
        loss=float(loss.item()),
        threshold=resolved_threshold,
    )


def average_inference_latency_ms(
    *,
    model: SentinelGAT,
    graph: HeteroData,
    device: str | torch.device,
    trials: int = 20,
    warmup_trials: int = 3,
) -> float:
    """Measure average milliseconds per transaction edge."""

    resolved_device = torch.device(device)
    graph_on_device = graph.to(resolved_device)
    model = model.to(resolved_device)
    model.eval()

    for _ in range(max(0, warmup_trials)):
        with torch.no_grad():
            model.predict_all_edges(graph_on_device)

    edge_count = max(1, sum(graph[edge_type].edge_index.shape[1] for edge_type in graph.edge_types))
    start = time.perf_counter()
    for _ in range(max(1, trials)):
        with torch.no_grad():
            model.predict_all_edges(graph_on_device)
    elapsed = time.perf_counter() - start
    return float((elapsed / max(1, trials)) * 1000.0 / edge_count)


def write_evaluation_report(
    report: EvaluationReport,
    output_path: str | Path,
) -> Path:
    """Persist an evaluation report as JSON."""

    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
    return resolved_path


def confusion_from_predictions(
    *,
    predictions: Tensor,
    labels: Tensor,
) -> ConfusionMatrix:
    """Compute binary confusion counts."""

    predictions = predictions.float().reshape(-1)
    labels = labels.float().reshape(-1)
    true_positive = int(((predictions == 1.0) & (labels == 1.0)).sum().item())
    false_positive = int(((predictions == 1.0) & (labels == 0.0)).sum().item())
    true_negative = int(((predictions == 0.0) & (labels == 0.0)).sum().item())
    false_negative = int(((predictions == 0.0) & (labels == 1.0)).sum().item())
    return ConfusionMatrix(
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def precision_from_confusion(confusion: ConfusionMatrix) -> float:
    denominator = confusion.true_positive + confusion.false_positive
    return confusion.true_positive / denominator if denominator else 0.0


def recall_from_confusion(confusion: ConfusionMatrix) -> float:
    denominator = confusion.true_positive + confusion.false_negative
    return confusion.true_positive / denominator if denominator else 0.0


def f1_from_precision_recall(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator else 0.0


def false_positive_rate_from_confusion(confusion: ConfusionMatrix) -> float:
    denominator = confusion.false_positive + confusion.true_negative
    return confusion.false_positive / denominator if denominator else 0.0


def binary_roc_auc(
    *,
    probabilities: Tensor,
    labels: Tensor,
) -> float:
    """Compute binary ROC-AUC without external metric dependencies."""

    probabilities = probabilities.detach().cpu().reshape(-1)
    labels = labels.detach().cpu().reshape(-1)

    positive_count = int((labels == 1.0).sum().item())
    negative_count = int((labels == 0.0).sum().item())
    if positive_count == 0 or negative_count == 0:
        return 0.0

    sorted_indices = torch.argsort(probabilities, descending=True)
    sorted_labels = labels[sorted_indices]
    true_positive = torch.cumsum(sorted_labels == 1.0, dim=0).float()
    false_positive = torch.cumsum(sorted_labels == 0.0, dim=0).float()

    tpr = torch.cat([torch.tensor([0.0]), true_positive / positive_count, torch.tensor([1.0])])
    fpr = torch.cat([torch.tensor([0.0]), false_positive / negative_count, torch.tensor([1.0])])
    area = torch.trapz(tpr, fpr)
    return float(area.item())
