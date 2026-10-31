"""Model components for the figure-faithful MS²F ablation study.

The paper figure has two graph-level readouts over the same PCA-reduced node
features: a pure semantic readout before message passing and a
structural-semantic readout after role-aware GATv2 encoding.  This module keeps
that distinction explicit so that each ablation removes exactly one claimed
component.
"""

from __future__ import annotations

from typing import Final

import torch
from torch_geometric.nn import GATv2Conv, global_mean_pool


ABLATION_VARIANTS: Final[tuple[str, ...]] = (
    "full",
    "semantic_only",
    "structural_only",
    "no_role_aware",
    "additive_fusion",
)


def global_max_pool_native(
    x: torch.Tensor,
    batch_index: torch.Tensor,
    *,
    size: int,
) -> torch.Tensor:
    """Pool maxima without requiring an optional torch-scatter wheel."""

    if batch_index.numel() == 0:
        return x.new_zeros((size, x.size(-1)))
    pooled = x.new_full((size, x.size(-1)), float("-inf"))
    expanded_index = batch_index.view(-1, 1).expand(-1, x.size(-1))
    pooled.scatter_reduce_(0, expanded_index, x, reduce="amax", include_self=True)
    return torch.nan_to_num(pooled, neginf=0.0)


def graph_readout(
    x: torch.Tensor,
    batch_index: torch.Tensor,
    pooling: str,
    *,
    size: int,
) -> torch.Tensor:
    """Return one graph vector per input graph using the configured readout."""

    if pooling == "mean":
        return global_mean_pool(x, batch_index, size=size)
    if pooling == "max":
        return global_max_pool_native(x, batch_index, size=size)
    if pooling == "mean_max":
        return torch.cat(
            [
                global_mean_pool(x, batch_index, size=size),
                global_max_pool_native(x, batch_index, size=size),
            ],
            dim=-1,
        )
    raise ValueError(f"Unsupported pooling: {pooling}")


class MS2FGraphClassifier(torch.nn.Module):
    """Dual-readout multi-label classifier matching the supplied framework.

    ``full`` concatenates pure semantic and structural-semantic readouts.
    ``semantic_only`` removes GATv2, ``structural_only`` removes the bypass
    readout, ``no_role_aware`` keeps both branches but removes node/edge role
    and relation-type embeddings, and ``additive_fusion`` replaces
    concatenation by element-wise averaging.
    """

    def __init__(
        self,
        *,
        variant: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        num_relations: int,
        num_edge_roles: int,
        dropout: float,
        pooling: str,
        heads: int,
    ) -> None:
        super().__init__()
        if variant not in ABLATION_VARIANTS:
            raise ValueError(f"Unsupported ablation variant: {variant}")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.variant = variant
        self.pooling = pooling
        self.use_semantic_branch = variant != "structural_only"
        self.use_structural_branch = variant != "semantic_only"
        self.role_aware = variant != "no_role_aware"
        self.additive_fusion = variant == "additive_fusion"

        self.input_projection = torch.nn.Linear(in_dim, hidden_dim)
        self.node_role_embedding = torch.nn.Embedding(2, hidden_dim)
        self.node_relation_embedding = torch.nn.Embedding(num_relations, hidden_dim)
        self.edge_role_embedding = torch.nn.Embedding(num_edge_roles, hidden_dim)
        self.edge_relation_embedding = torch.nn.Embedding(num_relations, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        if self.use_structural_branch:
            for _ in range(num_layers):
                self.convs.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        edge_dim=hidden_dim if self.role_aware else None,
                        add_self_loops=False,
                    )
                )
                self.norms.append(torch.nn.LayerNorm(hidden_dim))

        readout_dim = hidden_dim * 2 if pooling == "mean_max" else hidden_dim
        if self.use_semantic_branch and self.use_structural_branch and not self.additive_fusion:
            classifier_in = readout_dim * 2
        else:
            classifier_in = readout_dim
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(classifier_in, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def encode_branches(
        self, batch
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return graph-level semantic and structural-semantic embeddings."""

        base = torch.relu(self.input_projection(batch.x))
        num_graphs = int(batch.num_graphs)
        semantic_graph = None
        if self.use_semantic_branch:
            semantic_graph = graph_readout(
                base,
                batch.batch,
                self.pooling,
                size=num_graphs,
            )

        structural_graph = None
        if self.use_structural_branch:
            x = base
            edge_attr = None
            if self.role_aware:
                x = (
                    x
                    + self.node_role_embedding(batch.node_role)
                    + self.node_relation_embedding(batch.node_relation_type)
                )
                edge_attr = (
                    self.edge_role_embedding(batch.edge_role)
                    + self.edge_relation_embedding(batch.edge_relation_type)
                )
            for conv, norm in zip(self.convs, self.norms):
                residual = x
                if edge_attr is None:
                    updated = conv(x, batch.edge_index)
                else:
                    updated = conv(x, batch.edge_index, edge_attr=edge_attr)
                x = norm(residual + self.dropout(torch.relu(updated)))
            structural_graph = graph_readout(
                x,
                batch.batch,
                self.pooling,
                size=num_graphs,
            )
        return semantic_graph, structural_graph

    def forward(self, batch) -> torch.Tensor:
        semantic_graph, structural_graph = self.encode_branches(batch)
        if semantic_graph is None:
            fused = structural_graph
        elif structural_graph is None:
            fused = semantic_graph
        elif self.additive_fusion:
            fused = 0.5 * (semantic_graph + structural_graph)
        else:
            fused = torch.cat([semantic_graph, structural_graph], dim=-1)
        if fused is None:
            raise RuntimeError("At least one readout branch must be active")
        return self.classifier(fused)
