import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from data_utils import x_map, e_map


class AtomEdgeEmbedder(nn.Module):
    """
    Embed categorical node features (9 fields) and edge features (3 fields).
    The input data.x is shape [num_nodes, 9] with integer indices.
    The input data.edge_attr is shape [num_edges, 3] with integer indices.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Node categorical fields in order from the challenge PDF and data_utils.py
        node_field_sizes = [
            len(x_map["atomic_num"]),
            len(x_map["chirality"]),
            len(x_map["degree"]),
            len(x_map["formal_charge"]),
            len(x_map["num_hs"]),
            len(x_map["num_radical_electrons"]),
            len(x_map["hybridization"]),
            len(x_map["is_aromatic"]),
            len(x_map["is_in_ring"]),
        ]
        self.node_embs = nn.ModuleList([nn.Embedding(s, hidden_dim) for s in node_field_sizes])

        # Edge categorical fields: bond_type, stereo, is_conjugated
        edge_field_sizes = [
            len(e_map["bond_type"]),
            len(e_map["stereo"]),
            len(e_map["is_conjugated"]),
        ]
        self.edge_embs = nn.ModuleList([nn.Embedding(s, hidden_dim) for s in edge_field_sizes])

    def embed_nodes(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 9]
        out = 0
        for i, emb in enumerate(self.node_embs):
            out = out + emb(x[:, i])
        return out  # [N, hidden_dim]

    def embed_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        # edge_attr: [E, 3]
        out = 0
        for i, emb in enumerate(self.edge_embs):
            out = out + emb(edge_attr[:, i])
        return out  # [E, hidden_dim]


class MolGATEncoder(nn.Module):
    """
    GAT graph encoder producing:
      - node embeddings (for prefix conditioning)
      - pooled graph embedding (for retrieval alignment)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 768,   # match BERT embedding size (baseline embeddings)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.feats = AtomEdgeEmbedder(hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # We use edge_dim so GAT can condition attention on edge features.
        # First layer maps hidden_dim -> hidden_dim (with heads).
        in_dim = hidden_dim
        for l in range(num_layers):
            # For intermediate layers, use concat heads => output is heads*hidden_dim.
            # For final layer, we keep concat=True too, then reduce with a linear.
            conv = GATConv(
                in_channels=in_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
                edge_dim=hidden_dim,
                add_self_loops=True,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(heads * hidden_dim))
            in_dim = heads * hidden_dim

        # Reduce node dim back to hidden_dim for nicer prefix length/size
        self.node_reduce = nn.Linear(in_dim, hidden_dim)

        # Graph projection to BERT space (out_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch):
        """
        batch: torch_geometric.data.Batch with fields:
          x [N,9], edge_index [2,E], edge_attr [E,3], batch [N]
        returns:
          g: [B, out_dim] normalized
          h_node: [N, hidden_dim] node embeddings
        """
        x = batch.x.long()
        edge_attr = batch.edge_attr.long()

        h = self.feats.embed_nodes(x)               # [N, hidden_dim]
        e = self.feats.embed_edges(edge_attr)       # [E, hidden_dim]

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, batch.edge_index, e)        # [N, heads*hidden_dim]
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        h_node = self.node_reduce(h)                # [N, hidden_dim]

        # Graph pooling
        g_mean = global_mean_pool(h_node, batch.batch)  # [B, hidden_dim]
        g_max  = global_max_pool(h_node, batch.batch)   # [B, hidden_dim]
        g = torch.cat([g_mean, g_max], dim=-1)          # [B, 2*hidden_dim]
        g = self.graph_proj(g)                          # [B, out_dim]
        g = F.normalize(g, dim=-1)

        return g, h_node
