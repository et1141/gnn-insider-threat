import torch
from torch_geometric.data import Data

from certgnn.model import GCN


def test_stabilize_input_features_handles_extreme_values():
    x = torch.tensor(
        [
            [0.0, 1.0, 10.0, 1e8],
            [float("nan"), float("inf"), -5.0, 42.0],
        ],
        dtype=torch.float32,
    )
    out = GCN._stabilize_input_features(x)
    assert torch.isfinite(out).all()


def test_gcn_forward_stays_finite_with_large_feature_scale():
    gcn = GCN(num_node_features=54, hidden_dim=16)
    # 4-node toy graph with large count-like values in every feature.
    x = torch.full((4, 54), 5_000_000.0, dtype=torch.float32)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
        dtype=torch.long,
    )
    data = Data(x=x, edge_index=edge_index)
    out = gcn(data)
    assert torch.isfinite(out).all()
