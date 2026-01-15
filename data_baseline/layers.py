import torch
import torch.nn as nn
import math

# --- MODIFICATION 1 : AJOUT DE L'ENCODEUR ATOMIQUE ---
# Ordre : [AtomicNum, Chirality, Degree, Charge, NumHs, Radical, Hybrid, Aromatic, Ring]
ATOM_FEATURE_DIMS = [119, 9, 11, 12, 9, 5, 8, 2, 2]

class AtomFeatureEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList()
        for num_classes in ATOM_FEATURE_DIMS:
            self.embeddings.append(nn.Embedding(num_classes, emb_dim))

    def forward(self, x):
        out = 0
        for i, layer in enumerate(self.embeddings):
            col = x[:, i].long()
            out = out + layer(col)
        return out

class GeometricInputLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x, geo_features, neighbor_indices):
        """
        x: [N, Dim] (Features atomiques)
        geo_features: [N, K, 5] -> [Dist, SinPhi, CosPhi, SinTheta, CosTheta]
        """
                
        # 1. Masquage et Récupération (Standard)
        mask = (neighbor_indices != -1)
        safe_indices = neighbor_indices.clone()
        safe_indices[~mask] = 0
        
        # [N, K, Dim]
        neighbor_feats = x[safe_indices]

        # 2. Décomposition Géométrique
        dist      = geo_features[:, :, 0].unsqueeze(-1) # [N, K, 1]
        sin_phi   = geo_features[:, :, 1].unsqueeze(-1)
        cos_phi   = geo_features[:, :, 2].unsqueeze(-1)
        sin_theta = geo_features[:, :, 3].unsqueeze(-1)
        cos_theta = geo_features[:, :, 4].unsqueeze(-1)

        # 3. Calcul du Vecteur Unitaire (Direction pure)
        u_x = sin_theta * cos_phi 
        u_y = sin_theta * sin_phi 
        u_z = cos_theta           

        # 4. Calcul du Facteur d'échelle (Norme)
        scale = 1.0 / (dist + 1e-6)

        # 5. Projection et Atténuation
        feat_x = neighbor_feats * u_x * scale
        feat_y = neighbor_feats * u_y * scale
        feat_z = neighbor_feats * u_z * scale

        # 6. Concaténation
        # Sortie : [N, K, 3 * Dim]
        out = torch.cat([feat_x, feat_y, feat_z], dim=-1)
        
        return out * mask.unsqueeze(-1)
    
class GeometricFeatureProjector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # On passe de 3*Dim (X+Y+Z) à Dim
        assert input_dim % 3 == 0
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 3),
            nn.LayerNorm(input_dim // 3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.proj(x)

class LocalNeighborAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        hidden_dim = 4 * input_dim
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, neighbor_indices):
        key_padding_mask = (neighbor_indices == -1)
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x * (~key_padding_mask).unsqueeze(-1)

class LocalAggregation(nn.Module):
    def __init__(self, input_dim, max_k, output_dim):
        super().__init__()
        self.flatten_dim = input_dim * max_k
        self.proj = nn.Sequential(
            nn.Linear(self.flatten_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU()
        )

    def forward(self, x):
        return self.proj(x.view(x.size(0), -1))

class GlobalPositionalEncoder(nn.Module):
    def __init__(self, lap_dim, rw_dim, wl_vocab_size, hidden_dim):
        super().__init__()
        
        # Continu (Laplacien + RWSE)
        self.continuous_mlp = nn.Sequential(
            nn.Linear(lap_dim + rw_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Discret (WL)
        self.wl_embedding = nn.Embedding(wl_vocab_size, hidden_dim)
        self.wl_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pe_lap, pe_rw, pe_wl):
        # 1. Continue
        h_cont = self.continuous_mlp(torch.cat([pe_lap, pe_rw], dim=-1))
        
        # 2. Discret (Somme des embeddings WL sur les steps)
        h_wl = self.wl_proj(torch.sum(self.wl_embedding(pe_wl), dim=1))
        
        return h_cont + h_wl