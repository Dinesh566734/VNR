"""Sentinel-UPI graph attention network with edge-aware attention."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax

from src.data.paysim_loader import load_config


EdgeType = tuple[str, str, str]


class EdgeAwareGATConv(nn.Module):
    """Graph attention convolution that includes edge features in attention."""

    def __init__(
        self,
        src_in_channels: int,
        dst_in_channels: int,
        edge_in_channels: int,
        out_channels: int,
        heads: int,
        dropout: float,
        negative_slope: float,
        use_edge_features_in_attention: bool = True,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.use_edge_features_in_attention = use_edge_features_in_attention

        self.src_projection = nn.Linear(src_in_channels, heads * out_channels, bias=False)
        self.dst_projection = nn.Linear(dst_in_channels, heads * out_channels, bias=False)
        self.edge_projection = nn.Linear(edge_in_channels, heads * out_channels, bias=False)
        self.attention_vector = nn.Parameter(
            torch.empty(1, heads, out_channels * 3)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.src_projection.weight)
        nn.init.xavier_uniform_(self.dst_projection.weight)
        nn.init.xavier_uniform_(self.edge_projection.weight)
        nn.init.xavier_uniform_(self.attention_vector)

    def forward(
        self,
        x_src: Tensor,
        x_dst: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if edge_index.numel() == 0:
            empty_out = torch.zeros(
                (x_dst.shape[0], self.heads * self.out_channels),
                device=x_dst.device,
                dtype=x_dst.dtype,
            )
            empty_attention = torch.empty(
                (0, self.heads),
                device=x_dst.device,
                dtype=x_dst.dtype,
            )
            return empty_out, empty_attention

        src_index, dst_index = edge_index[0], edge_index[1]
        src_projected = self.src_projection(x_src).view(-1, self.heads, self.out_channels)
        dst_projected = self.dst_projection(x_dst).view(-1, self.heads, self.out_channels)
        if self.use_edge_features_in_attention:
            edge_projected = self.edge_projection(edge_attr).view(-1, self.heads, self.out_channels)
        else:
            edge_projected = torch.zeros(
                (edge_attr.shape[0], self.heads, self.out_channels),
                device=edge_attr.device,
                dtype=edge_attr.dtype,
            )

        src_messages = src_projected[src_index]
        dst_messages = dst_projected[dst_index]
        attention_input = torch.cat(
            [dst_messages, src_messages, edge_projected],
            dim=-1,
        )
        attention_scores = F.leaky_relu(
            (attention_input * self.attention_vector).sum(dim=-1),
            negative_slope=self.negative_slope,
        )
        attention_weights = softmax(
            attention_scores,
            index=dst_index,
            num_nodes=x_dst.shape[0],
        )
        dropped_attention = F.dropout(
            attention_weights,
            p=self.dropout,
            training=self.training,
        )

        messages = src_messages * dropped_attention.unsqueeze(-1)
        aggregated = torch.zeros(
            (x_dst.shape[0], self.heads, self.out_channels),
            device=x_dst.device,
            dtype=x_dst.dtype,
        )
        aggregated.index_add_(0, dst_index, messages)
        return aggregated.reshape(x_dst.shape[0], self.heads * self.out_channels), attention_weights


class SentinelGAT(nn.Module):
    """Two-layer edge-aware graph attention network for edge fraud scoring."""

    def __init__(
        self,
        *,
        edge_types: list[EdgeType] | tuple[EdgeType, ...],
        node_input_dims: dict[str, int],
        edge_input_dims: dict[EdgeType, int],
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.3,
        attention_negative_slope: float = 0.2,
        classifier_hidden_dims: tuple[int, int] = (128, 64),
        num_layers: int = 2,
        use_edge_features_in_attention: bool = True,
    ) -> None:
        super().__init__()
        self.edge_types = tuple(edge_types)
        self.node_types = tuple(node_input_dims.keys())
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.embedding_dim = hidden_channels * heads
        self.attention_negative_slope = attention_negative_slope
        self.edge_input_dims = dict(edge_input_dims)
        self.num_layers = num_layers
        self.use_edge_features_in_attention = use_edge_features_in_attention

        if num_layers < 1:
            raise ValueError("SentinelGAT requires at least one attention layer.")

        distinct_edge_dims = {edge_input_dims[edge_type] for edge_type in self.edge_types}
        if len(distinct_edge_dims) != 1:
            raise ValueError("All relations must share the same edge feature dimension.")
        self.edge_feature_dim = distinct_edge_dims.pop()

        self.layers = nn.ModuleList()
        self.skip_projections = nn.ModuleList()

        current_input_dims = dict(node_input_dims)
        for _ in range(num_layers):
            convs = nn.ModuleDict()
            for edge_type in self.edge_types:
                key = self._edge_type_key(edge_type)
                src_type, _, dst_type = edge_type
                convs[key] = EdgeAwareGATConv(
                    src_in_channels=current_input_dims[src_type],
                    dst_in_channels=current_input_dims[dst_type],
                    edge_in_channels=edge_input_dims[edge_type],
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    negative_slope=attention_negative_slope,
                    use_edge_features_in_attention=use_edge_features_in_attention,
                )
            self.layers.append(convs)
            self.skip_projections.append(
                nn.ModuleDict(
                    {
                        node_type: nn.Linear(
                            current_input_dims[node_type],
                            self.embedding_dim,
                            bias=False,
                        )
                        for node_type in self.node_types
                    }
                )
            )
            current_input_dims = {node_type: self.embedding_dim for node_type in self.node_types}

        first_hidden, second_hidden = classifier_hidden_dims
        classifier_input_dim = self.embedding_dim * 2 + self.edge_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, first_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    @classmethod
    def from_graph(
        cls,
        graph: HeteroData,
        config_path: str | None = None,
        **overrides: Any,
    ) -> "SentinelGAT":
        config = load_config(config_path)
        classifier_dims = tuple(config["model"]["classifier_hidden_dims"])
        if len(classifier_dims) != 2:
            raise ValueError("model.classifier_hidden_dims must contain exactly two values.")

        node_input_dims = {
            node_type: int(graph[node_type].x.shape[1])
            for node_type in graph.node_types
        }
        edge_input_dims = {
            edge_type: int(graph[edge_type].edge_attr.shape[1])
            for edge_type in graph.edge_types
        }
        constructor_args: dict[str, Any] = {
            "edge_types": graph.edge_types,
            "node_input_dims": node_input_dims,
            "edge_input_dims": edge_input_dims,
            "hidden_channels": int(config["model"]["hidden_channels"]),
            "heads": int(config["model"]["heads"]),
            "dropout": float(config["model"]["dropout"]),
            "attention_negative_slope": float(config["model"]["attention_negative_slope"]),
            "classifier_hidden_dims": classifier_dims,
        }
        constructor_args.update(overrides)
        return cls(
            **constructor_args,
        )

    def forward(
        self,
        graph: HeteroData,
        *,
        return_attention_weights: bool = False,
        return_logits: bool = False,
    ) -> dict[EdgeType, Tensor] | tuple[dict[EdgeType, Tensor], dict[EdgeType, Tensor]]:
        node_embeddings, attention_weights = self.encode(
            graph,
            return_attention_weights=return_attention_weights,
        )
        edge_outputs = self._score_edges(
            graph,
            node_embeddings=node_embeddings,
            return_logits=return_logits,
        )
        if return_attention_weights:
            return edge_outputs, attention_weights
        return edge_outputs

    def predict_all_edges(
        self,
        graph: HeteroData,
        *,
        return_logits: bool = False,
    ) -> Tensor:
        edge_outputs = self.forward(graph, return_logits=return_logits)
        tensors = [edge_outputs[edge_type] for edge_type in graph.edge_types]
        if not tensors:
            return torch.empty((0, 1), dtype=graph["user"].x.dtype)
        return torch.cat(tensors, dim=0)

    def encode(
        self,
        graph: HeteroData,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[dict[str, Tensor], dict[EdgeType, Tensor]]:
        x_dict = {
            node_type: graph[node_type].x.float()
            for node_type in graph.node_types
        }

        final_attention: dict[EdgeType, Tensor] = {}
        for layer_index, (convs, skip_projections) in enumerate(
            zip(self.layers, self.skip_projections, strict=True)
        ):
            x_dict, layer_attention = self._run_layer(
                x_dict=x_dict,
                graph=graph,
                convs=convs,
                skip_projections=skip_projections,
                capture_attention=return_attention_weights and layer_index == self.num_layers - 1,
            )
            if layer_attention:
                final_attention = layer_attention
        return x_dict, final_attention

    def _score_edges(
        self,
        graph: HeteroData,
        *,
        node_embeddings: dict[str, Tensor],
        return_logits: bool,
    ) -> dict[EdgeType, Tensor]:
        outputs: dict[EdgeType, Tensor] = {}
        for edge_type in self.edge_types:
            edge_store = graph[edge_type]
            if edge_store.edge_index.numel() == 0:
                outputs[edge_type] = torch.empty((0, 1), dtype=node_embeddings["user"].dtype)
                continue

            src_type, _, dst_type = edge_type
            src_embeddings = node_embeddings[src_type][edge_store.edge_index[0]]
            dst_embeddings = node_embeddings[dst_type][edge_store.edge_index[1]]
            classifier_input = torch.cat(
                [src_embeddings, dst_embeddings, edge_store.edge_attr.float()],
                dim=-1,
            )
            logits = self.classifier(classifier_input)
            outputs[edge_type] = logits if return_logits else torch.sigmoid(logits)
        return outputs

    def _run_layer(
        self,
        *,
        x_dict: dict[str, Tensor],
        graph: HeteroData,
        convs: nn.ModuleDict,
        skip_projections: nn.ModuleDict,
        capture_attention: bool,
    ) -> tuple[dict[str, Tensor], dict[EdgeType, Tensor]]:
        next_embeddings = {
            node_type: skip_projections[node_type](x_dict[node_type])
            for node_type in self.node_types
        }
        attention_weights: dict[EdgeType, Tensor] = {}

        for edge_type in self.edge_types:
            key = self._edge_type_key(edge_type)
            src_type, _, dst_type = edge_type
            aggregated, relation_attention = convs[key](
                x_src=x_dict[src_type],
                x_dst=x_dict[dst_type],
                edge_index=graph[edge_type].edge_index,
                edge_attr=graph[edge_type].edge_attr.float(),
            )
            next_embeddings[dst_type] = next_embeddings[dst_type] + aggregated
            if capture_attention:
                attention_weights[edge_type] = relation_attention

        for node_type in self.node_types:
            next_embeddings[node_type] = F.elu(next_embeddings[node_type])
            next_embeddings[node_type] = F.dropout(
                next_embeddings[node_type],
                p=self.dropout,
                training=self.training,
            )

        return next_embeddings, attention_weights

    @staticmethod
    def _edge_type_key(edge_type: EdgeType) -> str:
        return "__".join(edge_type)
