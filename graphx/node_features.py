"""
graphx/node_features.py

Build leakage-safe node features for a PyTorch Geometric graph.

The existing FeatureContext is the source of truth for feature computation.

For a given `as_of`:
    - only transactions with ts < as_of are visible
    - only observed transactions are used by graph features
    - labels are NOT used
    - the existing graph feature definitions are reused

This module does NOT:
    - train a GNN
    - create labels
    - modify FeatureContext
    - reconstruct trails
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import torch

from graphx.features import BASE_FEATURES, compute_graph_features
from ml.features import FeatureContext


def build_node_feature_matrix(
    ctx: FeatureContext,
    id_to_node: list[str],
    as_of: datetime,
    trail=None,
) -> torch.Tensor:
    """
    Build an as-of node feature matrix.

    Parameters
    ----------
    ctx:
        Existing FeatureContext.

    id_to_node:
        List where index -> account_id.

    as_of:
        Snapshot timestamp.

    trail:
        Optional existing money-flow trail.

    Returns
    -------
    torch.Tensor
        Shape:

            [num_nodes, len(BASE_FEATURES)]

        For the current graph this is:

            [num_nodes, 8]
    """

    rows = []

    for account_id in id_to_node:

        features = compute_graph_features(
            account_id=account_id,
            as_of=as_of,
            ctx=ctx,
            trail=trail,
        )

        rows.append(
            [
                float(features[name])
                for name in BASE_FEATURES
            ]
        )

    if not rows:
        return torch.empty(
            (0, len(BASE_FEATURES)),
            dtype=torch.float32,
        )

    X = np.asarray(rows, dtype=np.float32)

    return torch.from_numpy(X)


def attach_node_features(
    data,
    ctx: FeatureContext,
    id_to_node: list[str],
    as_of: datetime,
    trail=None,
):
    """
    Compute node features and attach them to a PyG Data object.

    Returns the same Data object with:

        data.x

    added.
    """

    X = build_node_feature_matrix(
        ctx=ctx,
        id_to_node=id_to_node,
        as_of=as_of,
        trail=trail,
    )

    data.x = X

    return data
    