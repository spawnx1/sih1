"""
Basic integrity checks for the PyG transaction graph.
"""

from ml.features import FeatureContext
from graphx.pyg_data import build_transaction_graph


def main():
    print("Loading FeatureContext...")
    ctx = FeatureContext()

    print("Building graph...")
    data, node_to_id, id_to_node = build_transaction_graph(ctx)

    print()
    print("========== GRAPH SUMMARY ==========")
    print("Nodes:", data.num_nodes)
    print("Edges:", data.edge_index.shape[1])
    print("edge_index shape:", tuple(data.edge_index.shape))
    print("edge_attr shape:", tuple(data.edge_attr.shape))
    print("edge_time shape:", tuple(data.edge_time.shape))
    print("Transaction IDs:", len(data.txn_id))

    print()
    print("========== INTEGRITY CHECKS ==========")

    # ------------------------------------------------------------
    # Check 1: node mapping
    # ------------------------------------------------------------

    assert len(node_to_id) == data.num_nodes
    assert len(id_to_node) == data.num_nodes

    print("Node mapping: OK")

    # ------------------------------------------------------------
    # Check 2: edge_index dimensions
    # ------------------------------------------------------------

    assert data.edge_index.dim() == 2
    assert data.edge_index.shape[0] == 2

    print("edge_index dimensions: OK")

    # ------------------------------------------------------------
    # Check 3: edge attributes
    # ------------------------------------------------------------

    assert data.edge_attr.shape[0] == data.edge_index.shape[1]

    print("edge_attr alignment: OK")

    # ------------------------------------------------------------
    # Check 4: timestamps
    # ------------------------------------------------------------

    assert data.edge_time.shape[0] == data.edge_index.shape[1]

    print("edge_time alignment: OK")

    # ------------------------------------------------------------
    # Check 5: transaction IDs
    # ------------------------------------------------------------

    assert len(data.txn_id) == data.edge_index.shape[1]

    print("transaction ID alignment: OK")

    # ------------------------------------------------------------
    # Check 6: node IDs are within range
    # ------------------------------------------------------------

    if data.edge_index.numel() > 0:
        assert int(data.edge_index.min()) >= 0
        assert int(data.edge_index.max()) < data.num_nodes

    print("Node index range: OK")

    # ------------------------------------------------------------
    # Check 7: no NaN / infinity in edge attributes
    # ------------------------------------------------------------

    assert not data.edge_attr.isnan().any()
    assert not data.edge_attr.isinf().any()

    print("Edge attributes finite: OK")

    print()
    print("========== RESULT ==========")
    print("GRAPH INTEGRITY: PASS")


if __name__ == "__main__":
    main()