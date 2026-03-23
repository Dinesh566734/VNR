"""Ablation studies and baseline benchmarks for Sentinel-UPI."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv

from src.data.paysim_loader import PROJECT_ROOT, load_config
from src.models.gat import SentinelGAT
from .eval import (
    EvaluationReport,
    binary_roc_auc,
    confusion_from_predictions,
    evaluate_test_graph,
    f1_from_precision_recall,
    false_positive_rate_from_confusion,
    precision_from_confusion,
    recall_from_confusion,
)
from .train import TrainingArtifacts, train_model

try:  # pragma: no cover - import availability depends on environment
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    LogisticRegression = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    RandomForestClassifier = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]

try:  # pragma: no cover - import availability depends on environment
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]


TrainFn = Callable[..., TrainingArtifacts]
EdgeType = tuple[str, str, str]


@dataclass
class AblationResult:
    experiment: str
    variant: str
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    false_positive_rate: float
    monthly_loss_inr: float
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AblationStudyReport:
    results: list[AblationResult]
    markdown_table: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "markdown_table": self.markdown_table,
        }


@dataclass
class BenchmarkResult:
    model_name: str
    model_type: str
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    latency_ms: float
    threshold: float
    monthly_loss_inr: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    results: list[BenchmarkResult]
    markdown_table: str
    output_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "markdown_table": self.markdown_table,
            "output_path": str(self.output_path) if self.output_path else None,
        }


class GraphSAGEEdgeClassifier(nn.Module):
    """Homogeneous GraphSAGE edge classifier baseline."""

    def __init__(
        self,
        *,
        input_channels: int,
        edge_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("GraphSAGE baseline requires at least one layer.")

        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_channels, hidden_channels))
        for _ in range(max(0, num_layers - 1)):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        classifier_input_dim = hidden_channels * 2 + edge_channels
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1),
        )

    def encode(self, data: Data) -> Tensor:
        x = data.x.float()
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, data: Data, *, return_logits: bool = False) -> Tensor:
        embeddings = self.encode(data)
        src_embeddings = embeddings[data.edge_index[0]]
        dst_embeddings = embeddings[data.edge_index[1]]
        logits = self.classifier(
            torch.cat([src_embeddings, dst_embeddings, data.edge_attr.float()], dim=-1)
        ).reshape(-1)
        return logits if return_logits else torch.sigmoid(logits)


def run_ablation_studies(
    *,
    train_graph: HeteroData,
    val_graph: HeteroData,
    cluster_assignments: dict[str, Any] | None = None,
    config_path: str | None = None,
    device: str | torch.device | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    verbose: bool = False,
    attention_heads: Iterable[int] = (1, 2, 4, 8),
    depths: Iterable[int] = (1, 2, 3),
    latency_trials: int = 5,
    train_fn: TrainFn = train_model,
) -> AblationStudyReport:
    """Run the plan's ablation experiments and return a markdown summary."""

    resolved_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    results: list[AblationResult] = []

    experiment_specs = [
        {
            "experiment": "Drop edge features from attention",
            "variants": [("disabled", {"use_edge_features_in_attention": False}, None)],
        },
        {
            "experiment": "CrossEntropy instead of Focal Loss",
            "variants": [("bce_logits", {}, nn.BCEWithLogitsLoss())],
        },
        {
            "experiment": "Attention heads",
            "variants": [(str(heads), {"heads": int(heads)}, None) for heads in attention_heads],
        },
        {
            "experiment": "GAT depth",
            "variants": [(str(depth), {"num_layers": int(depth)}, None) for depth in depths],
        },
    ]

    for spec in experiment_specs:
        for variant_name, model_kwargs, criterion in spec["variants"]:
            artifacts = train_fn(
                train_graph=train_graph,
                val_graph=val_graph,
                cluster_assignments=cluster_assignments,
                config_path=config_path,
                device=resolved_device,
                max_epochs=max_epochs,
                patience=patience,
                model_kwargs=model_kwargs,
                criterion=criterion,
                save_checkpoint=False,
                verbose=verbose,
            )
            evaluation = evaluate_test_graph(
                model=artifacts.model,
                graph=val_graph,
                config_path=config_path,
                device=resolved_device,
                latency_trials=latency_trials,
                warmup_trials=1,
            )
            results.append(_result_from_evaluation(spec["experiment"], variant_name, evaluation))

    markdown_table = ablation_results_to_markdown(results)
    return AblationStudyReport(results=results, markdown_table=markdown_table)


def run_baseline_benchmarks(
    *,
    train_graph: HeteroData,
    val_graph: HeteroData,
    test_graph: HeteroData,
    config_path: str | None = None,
    output_path: str | Path | None = None,
    device: str | torch.device | None = None,
    sentinel_evaluation: EvaluationReport | None = None,
    train_fn: TrainFn = train_model,
    max_sentinel_epochs: int | None = None,
    patience: int | None = None,
    verbose: bool = False,
) -> BenchmarkReport:
    """Train and evaluate tabular, graph, and Sentinel baselines on the same split."""

    _require_benchmark_dependencies()
    config = load_config(config_path)
    benchmark_config = config.get("benchmark", {})
    threshold_candidates = _threshold_candidates(config=benchmark_config)

    train_x, train_y = _tabular_edge_dataset(train_graph)
    val_x, val_y = _tabular_edge_dataset(val_graph)
    test_x, test_y = _tabular_edge_dataset(test_graph)
    _validate_binary_labels(train_y, split_name="train")
    _validate_binary_labels(val_y, split_name="validation")
    _validate_binary_labels(test_y, split_name="test")

    results = [
        _run_logistic_regression_benchmark(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
            config=benchmark_config.get("logistic_regression", {}),
            threshold_candidates=threshold_candidates,
        ),
        _run_random_forest_benchmark(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
            config=benchmark_config.get("random_forest", {}),
            threshold_candidates=threshold_candidates,
        ),
        _run_xgboost_benchmark(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
            config=benchmark_config.get("xgboost", {}),
            threshold_candidates=threshold_candidates,
        ),
        _run_graphsage_benchmark(
            train_graph=train_graph,
            val_graph=val_graph,
            test_graph=test_graph,
            device=device,
            config=benchmark_config.get("graphsage", {}),
            threshold_candidates=threshold_candidates,
            verbose=verbose,
        ),
    ]

    cluster_assignments = _benchmark_cluster_assignments(train_graph=train_graph, config=config)
    if sentinel_evaluation is None:
        sentinel_artifacts = train_fn(
            train_graph=train_graph,
            val_graph=val_graph,
            cluster_assignments=cluster_assignments,
            config_path=config_path,
            device=device,
            max_epochs=max_sentinel_epochs,
            patience=patience,
            save_checkpoint=False,
            verbose=verbose,
        )
        sentinel_evaluation = evaluate_test_graph(
            model=sentinel_artifacts.model,
            graph=test_graph,
            config_path=config_path,
            device=device,
            latency_trials=int(benchmark_config.get("latency_trials", 5)),
            warmup_trials=1,
        )

    results.append(
        BenchmarkResult(
            model_name="Sentinel-UPI",
            model_type="Graph (GAT)",
            precision=sentinel_evaluation.precision,
            recall=sentinel_evaluation.recall,
            f1_score=sentinel_evaluation.f1_score,
            auc_roc=sentinel_evaluation.auc_roc,
            latency_ms=sentinel_evaluation.average_latency_ms,
            threshold=sentinel_evaluation.threshold,
            monthly_loss_inr=sentinel_evaluation.monthly_loss_inr,
        )
    )

    markdown_table = benchmark_results_to_markdown(results)
    resolved_output: Path | None = None
    if output_path is not None:
        resolved_output = Path(output_path)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(markdown_table + "\n", encoding="utf-8")
        resolved_output.with_suffix(".json").write_text(
            json.dumps([result.to_dict() for result in results], indent=2),
            encoding="utf-8",
        )

    return BenchmarkReport(results=results, markdown_table=markdown_table, output_path=resolved_output)


def run_benchmarks_from_processed_artifacts(
    *,
    processed_dir: str | Path | None = None,
    config_path: str | None = None,
    output_path: str | Path | None = None,
    device: str | torch.device | None = None,
    max_sentinel_epochs: int | None = None,
    patience: int | None = None,
    verbose: bool = False,
) -> BenchmarkReport:
    """Load saved graph splits and checkpoint, then write the benchmark markdown report."""

    run_dir = _resolve_processed_dir(processed_dir)
    train_graph = torch.load(run_dir / "graph_train.pt", weights_only=False)
    val_graph = torch.load(run_dir / "graph_val.pt", weights_only=False)
    test_graph = torch.load(run_dir / "graph_test.pt", weights_only=False)

    sentinel_evaluation = None
    checkpoint_path = run_dir / "sentinel_gat_best.pt"
    if checkpoint_path.is_file():
        model = SentinelGAT.from_graph(train_graph, config_path=config_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        sentinel_evaluation = evaluate_test_graph(
            model=model,
            graph=test_graph,
            config_path=config_path,
            device=device,
            latency_trials=5,
            warmup_trials=1,
        )

    resolved_output = (
        Path(output_path)
        if output_path is not None
        else PROJECT_ROOT / "results" / "benchmark.md"
    )
    return run_baseline_benchmarks(
        train_graph=train_graph,
        val_graph=val_graph,
        test_graph=test_graph,
        config_path=config_path,
        output_path=resolved_output,
        device=device,
        sentinel_evaluation=sentinel_evaluation,
        max_sentinel_epochs=max_sentinel_epochs,
        patience=patience,
        verbose=verbose,
    )


def ablation_results_to_markdown(results: list[AblationResult]) -> str:
    """Render ablation results as a markdown table."""

    header = (
        "| Experiment | Variant | Precision | Recall | F1 | AUC-ROC | FPR | Monthly Loss (INR) | Latency (ms) |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [
        (
            f"| {result.experiment} | {result.variant} | "
            f"{result.precision:.3f} | {result.recall:.3f} | {result.f1_score:.3f} | "
            f"{result.auc_roc:.3f} | {result.false_positive_rate:.3f} | "
            f"{result.monthly_loss_inr:.2f} | {result.latency_ms:.3f} |"
        )
        for result in results
    ]
    return "\n".join([header, *rows])


def benchmark_results_to_markdown(results: list[BenchmarkResult]) -> str:
    """Render baseline benchmark results as the requested markdown table."""

    header = (
        "| Model | Type | Precision | Recall | F1 | AUC-ROC | Latency (ms) |\n"
        "|---|---|---:|---:|---:|---:|---:|"
    )
    rows = [
        (
            f"| {result.model_name} | {result.model_type} | "
            f"{result.precision:.3f} | {result.recall:.3f} | {result.f1_score:.3f} | "
            f"{result.auc_roc:.3f} | {result.latency_ms:.3f} |"
        )
        for result in results
    ]
    return "\n".join([header, *rows])


def _run_logistic_regression_benchmark(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    config: dict[str, Any],
    threshold_candidates: np.ndarray,
) -> BenchmarkResult:
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=int(config.get("max_iter", 400)),
                    class_weight="balanced",
                    random_state=int(config.get("random_state", 17)),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    estimator.fit(train_x, train_y)
    return _sklearn_benchmark_result(
        model_name="Logistic Reg.",
        model_type="Statistical",
        estimator=estimator,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        threshold_candidates=threshold_candidates,
    )


def _run_random_forest_benchmark(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    config: dict[str, Any],
    threshold_candidates: np.ndarray,
) -> BenchmarkResult:
    estimator = RandomForestClassifier(
        n_estimators=int(config.get("n_estimators", 160)),
        max_depth=int(config.get("max_depth", 12)),
        min_samples_leaf=int(config.get("min_samples_leaf", 2)),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=int(config.get("random_state", 17)),
    )
    estimator.fit(train_x, train_y)
    return _sklearn_benchmark_result(
        model_name="Random Forest",
        model_type="Ensemble",
        estimator=estimator,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        threshold_candidates=threshold_candidates,
    )


def _run_xgboost_benchmark(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    config: dict[str, Any],
    threshold_candidates: np.ndarray,
) -> BenchmarkResult:
    positives = max(1, int(train_y.sum()))
    negatives = max(1, int(train_y.shape[0] - positives))
    estimator = XGBClassifier(
        n_estimators=int(config.get("n_estimators", 160)),
        max_depth=int(config.get("max_depth", 6)),
        learning_rate=float(config.get("learning_rate", 0.05)),
        subsample=float(config.get("subsample", 0.9)),
        colsample_bytree=float(config.get("colsample_bytree", 0.9)),
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float(negatives / positives),
        random_state=int(config.get("random_state", 17)),
        tree_method="hist",
        n_jobs=0,
    )
    estimator.fit(train_x, train_y)
    return _sklearn_benchmark_result(
        model_name="XGBoost",
        model_type="Boosting",
        estimator=estimator,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        threshold_candidates=threshold_candidates,
    )


def _run_graphsage_benchmark(
    *,
    train_graph: HeteroData,
    val_graph: HeteroData,
    test_graph: HeteroData,
    device: str | torch.device | None,
    config: dict[str, Any],
    threshold_candidates: np.ndarray,
    verbose: bool,
) -> BenchmarkResult:
    train_data = _hetero_graph_to_homogeneous(train_graph)
    val_data = _hetero_graph_to_homogeneous(val_graph)
    test_data = _hetero_graph_to_homogeneous(test_graph)
    resolved_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = GraphSAGEEdgeClassifier(
        input_channels=int(train_data.x.shape[1]),
        edge_channels=int(train_data.edge_attr.shape[1]),
        hidden_channels=int(config.get("hidden_channels", 64)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(resolved_device)
    train_on_device = train_data.to(resolved_device)
    val_on_device = val_data.to(resolved_device)
    test_on_device = test_data.to(resolved_device)

    positives = float(train_on_device.y.sum().item())
    negatives = float(train_on_device.y.numel() - positives)
    pos_weight = torch.tensor([negatives / max(1.0, positives)], device=resolved_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.003)),
        weight_decay=float(config.get("weight_decay", 5e-4)),
    )

    best_state: dict[str, Tensor] | None = None
    best_threshold = 0.5
    best_val_f1 = float("-inf")
    patience_limit = int(config.get("early_stopping_patience", 5))
    patience_counter = 0

    for epoch in range(1, int(config.get("max_epochs", 25)) + 1):
        model.train()
        optimizer.zero_grad()
        train_logits = model(train_on_device, return_logits=True)
        train_loss = criterion(train_logits, train_on_device.y.float())
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probabilities = model(val_on_device).detach().cpu().numpy()
        threshold = _tune_threshold(val_probabilities, val_data.y.numpy(), threshold_candidates)
        val_metrics = _metrics_from_probabilities(
            model_name="GraphSAGE",
            model_type="Graph (GNN)",
            probabilities=val_probabilities,
            labels=val_data.y.numpy(),
            threshold=threshold,
            latency_ms=0.0,
        )

        if verbose:
            print(
                f"GraphSAGE epoch {epoch:03d} | loss={train_loss.item():.4f} | "
                f"val_f1={val_metrics.f1_score:.3f} | threshold={threshold:.2f}"
            )

        if val_metrics.f1_score > best_val_f1:
            best_val_f1 = val_metrics.f1_score
            best_threshold = threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(resolved_device).eval()
    with torch.no_grad():
        test_probabilities = model(test_on_device).detach().cpu().numpy()
    latency_ms = _measure_graph_model_latency_ms(model=model, data=test_on_device)
    return _metrics_from_probabilities(
        model_name="GraphSAGE",
        model_type="Graph (GNN)",
        probabilities=test_probabilities,
        labels=test_data.y.numpy(),
        threshold=best_threshold,
        latency_ms=latency_ms,
    )


def _sklearn_benchmark_result(
    *,
    model_name: str,
    model_type: str,
    estimator: Any,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    threshold_candidates: np.ndarray,
) -> BenchmarkResult:
    val_probabilities = estimator.predict_proba(val_x)[:, 1]
    threshold = _tune_threshold(val_probabilities, val_y, threshold_candidates)
    latency_ms = _measure_predict_proba_latency_ms(estimator=estimator, features=test_x)
    test_probabilities = estimator.predict_proba(test_x)[:, 1]
    return _metrics_from_probabilities(
        model_name=model_name,
        model_type=model_type,
        probabilities=test_probabilities,
        labels=test_y,
        threshold=threshold,
        latency_ms=latency_ms,
    )


def _metrics_from_probabilities(
    *,
    model_name: str,
    model_type: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    latency_ms: float,
) -> BenchmarkResult:
    probability_tensor = torch.tensor(probabilities, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    prediction_tensor = (probability_tensor >= float(threshold)).float()

    confusion = confusion_from_predictions(predictions=prediction_tensor, labels=label_tensor)
    precision = precision_from_confusion(confusion)
    recall = recall_from_confusion(confusion)
    f1_score = f1_from_precision_recall(precision, recall)
    auc_roc = binary_roc_auc(probabilities=probability_tensor, labels=label_tensor)
    false_positive_rate = false_positive_rate_from_confusion(confusion)
    monthly_loss_inr = float(confusion.false_positive * 200 + confusion.false_negative * 15000)

    return BenchmarkResult(
        model_name=model_name,
        model_type=model_type,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        auc_roc=auc_roc,
        latency_ms=latency_ms,
        threshold=float(threshold),
        monthly_loss_inr=monthly_loss_inr + false_positive_rate * 0.0,
    )


def _tabular_edge_dataset(graph: HeteroData) -> tuple[np.ndarray, np.ndarray]:
    features: list[Tensor] = []
    labels: list[Tensor] = []

    for edge_type in graph.edge_types:
        edge_store = graph[edge_type]
        if edge_store.edge_index.numel() == 0:
            continue
        src_embeddings = graph[edge_type[0]].x[edge_store.edge_index[0]].float()
        dst_embeddings = graph[edge_type[2]].x[edge_store.edge_index[1]].float()
        relation_one_hot = torch.tensor(
            [1.0, 0.0] if edge_type[2] == "user" else [0.0, 1.0],
            dtype=torch.float32,
        ).repeat(edge_store.edge_index.shape[1], 1)
        features.append(
            torch.cat([src_embeddings, dst_embeddings, edge_store.edge_attr.float(), relation_one_hot], dim=1)
        )
        labels.append(edge_store.y.float())

    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return (
        torch.cat(features, dim=0).cpu().numpy().astype(np.float32),
        torch.cat(labels, dim=0).cpu().numpy().astype(np.int64),
    )


def _hetero_graph_to_homogeneous(graph: HeteroData) -> Data:
    user_x = graph["user"].x.float()
    merchant_x = graph["merchant"].x.float()
    merchant_offset = user_x.shape[0]

    edge_index_parts: list[Tensor] = []
    edge_attr_parts: list[Tensor] = []
    label_parts: list[Tensor] = []

    for edge_type in graph.edge_types:
        edge_store = graph[edge_type]
        if edge_store.edge_index.numel() == 0:
            continue
        src_index = edge_store.edge_index[0]
        dst_index = edge_store.edge_index[1]
        if edge_type[2] == "merchant":
            dst_index = dst_index + merchant_offset
        relation_one_hot = torch.tensor(
            [1.0, 0.0] if edge_type[2] == "user" else [0.0, 1.0],
            dtype=torch.float32,
        ).repeat(edge_store.edge_index.shape[1], 1)
        edge_index_parts.append(torch.stack([src_index, dst_index], dim=0))
        edge_attr_parts.append(torch.cat([edge_store.edge_attr.float(), relation_one_hot], dim=1))
        label_parts.append(edge_store.y.float())

    if not edge_index_parts:
        raise ValueError("Graph benchmark requires at least one edge.")

    return Data(
        x=torch.cat([user_x, merchant_x], dim=0),
        edge_index=torch.cat(edge_index_parts, dim=1),
        edge_attr=torch.cat(edge_attr_parts, dim=0),
        y=torch.cat(label_parts, dim=0),
    )


def _measure_predict_proba_latency_ms(
    *,
    estimator: Any,
    features: np.ndarray,
    trials: int = 5,
) -> float:
    if features.shape[0] == 0:
        return 0.0
    estimator.predict_proba(features[: min(len(features), 256)])
    start = time.perf_counter()
    for _ in range(max(1, trials)):
        estimator.predict_proba(features)
    elapsed = time.perf_counter() - start
    return float((elapsed / max(1, trials)) * 1000.0 / max(1, features.shape[0]))


def _measure_graph_model_latency_ms(
    *,
    model: GraphSAGEEdgeClassifier,
    data: Data,
    trials: int = 5,
) -> float:
    if data.y.numel() == 0:
        return 0.0
    with torch.no_grad():
        model(data)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max(1, trials)):
            model(data)
    elapsed = time.perf_counter() - start
    return float((elapsed / max(1, trials)) * 1000.0 / max(1, int(data.y.numel())))


def _threshold_candidates(*, config: dict[str, Any]) -> np.ndarray:
    start = float(config.get("threshold_min", 0.10))
    end = float(config.get("threshold_max", 0.90))
    steps = int(config.get("threshold_steps", 17))
    return np.linspace(start, end, num=steps, dtype=np.float32)


def _tune_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold_candidates: np.ndarray,
) -> float:
    best_threshold = 0.5
    best_f1 = float("-inf")
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    probability_tensor = torch.tensor(probabilities, dtype=torch.float32)

    for threshold in threshold_candidates:
        predictions = (probability_tensor >= float(threshold)).float()
        confusion = confusion_from_predictions(predictions=predictions, labels=label_tensor)
        precision = precision_from_confusion(confusion)
        recall = recall_from_confusion(confusion)
        f1_score = f1_from_precision_recall(precision, recall)
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = float(threshold)
    return best_threshold


def _resolve_processed_dir(processed_dir: str | Path | None) -> Path:
    if processed_dir is not None:
        resolved = Path(processed_dir)
    else:
        resolved = PROJECT_ROOT / "data" / "processed" / "real_run_150k"
    if not resolved.is_dir():
        raise FileNotFoundError(f"Processed benchmark directory not found: {resolved}")
    return resolved


def _benchmark_cluster_assignments(
    *,
    train_graph: HeteroData,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    requested_clusters = int(config.get("partitioner", {}).get("num_clusters", 50))
    total_nodes = int(train_graph["user"].num_nodes) + int(train_graph["merchant"].num_nodes)
    if total_nodes >= requested_clusters:
        return None
    return {
        "user": torch.zeros(train_graph["user"].num_nodes, dtype=torch.long),
        "merchant": torch.zeros(train_graph["merchant"].num_nodes, dtype=torch.long),
        "method": "manual",
        "num_clusters": 1,
    }


def _validate_binary_labels(labels: np.ndarray, *, split_name: str) -> None:
    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError(
            f"Benchmark split '{split_name}' must contain both classes, found labels {unique.tolist()}."
        )


def _require_benchmark_dependencies() -> None:
    missing = []
    if LogisticRegression is None or Pipeline is None or RandomForestClassifier is None or StandardScaler is None:
        missing.append("scikit-learn")
    if XGBClassifier is None:
        missing.append("xgboost")
    if missing:
        raise ImportError(
            "Missing benchmark dependencies: " + ", ".join(missing)
        )


def _result_from_evaluation(
    experiment: str,
    variant: str,
    evaluation: EvaluationReport,
) -> AblationResult:
    return AblationResult(
        experiment=experiment,
        variant=variant,
        precision=evaluation.precision,
        recall=evaluation.recall,
        f1_score=evaluation.f1_score,
        auc_roc=evaluation.auc_roc,
        false_positive_rate=evaluation.false_positive_rate,
        monthly_loss_inr=evaluation.monthly_loss_inr,
        latency_ms=evaluation.average_latency_ms,
    )
