import torch
import torch.nn as nn


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        m = self.edge_mlp(h[src] + edge_attr)
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        return h + self.node_mlp(h + agg)


class FoldCast(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        self.embed = nn.Linear(32, hidden_dim)
        self.blocks = nn.ModuleList(
            [EquivariantBlock(hidden_dim) for _ in range(num_layers)]
        )
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, coords, residue_features, edge_index, edge_attr):
        h = self.embed(residue_features)
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)
        velocity = self.flow_head(h)
        return velocity
