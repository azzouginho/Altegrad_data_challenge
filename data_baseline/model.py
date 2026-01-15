import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from layers import AtomFeatureEncoder, GeometricInputLayer, GeometricFeatureProjector, LocalNeighborAttention, LocalAggregation, GlobalPositionalEncoder

class FullEncoderModel(nn.Module):
    def __init__(self, 
                 lap_dim=8, 
                 rw_dim=16, 
                 wl_vocab_size=50000, 
                 max_k=11, 
                 hidden_dim=256, 
                 num_heads=8, 
                 num_layers=4,
                 output_dim=768):
        super().__init__()
        
        self.atom_encoder = AtomFeatureEncoder(hidden_dim)

        # --- BRANCHE A : LOCALE (Chimie + Géométrie 3D) ---
        
        # 1. Couche Géométrique Pure
        self.geo_input = GeometricInputLayer(feature_dim=hidden_dim)
        
        # 2. Projecteur de Fusion (Réduction de dimension)
        self.geo_projector = GeometricFeatureProjector(input_dim=hidden_dim * 3)
        
        # Le reste de la branche locale (Attention)
        self.layers = nn.ModuleList([
            LocalNeighborAttention(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.local_agg = LocalAggregation(hidden_dim, max_k, hidden_dim)
        
        # --- BRANCHE B : GLOBALE (Structurelle) ---
        self.global_encoder = GlobalPositionalEncoder(lap_dim, rw_dim, wl_vocab_size, hidden_dim)
        
        # --- FUSION & SORTIE ---
        fusion_dim = hidden_dim * 3
        
        self.final_mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, data, sorted_neighbors, geo_features): # <-- Nouvel argument ici
        # 1. --- Encodage Atomique ---
        h_atom = self.atom_encoder(data.x) # [N, Hidden]

        # 2. --- Branche Locale ---
        x_raw = self.geo_input(h_atom, geo_features, sorted_neighbors)
        
        # B. Réduction & Normalisation -> Sortie [N, K, Hidden]
        x = self.geo_projector(x_raw)
        
        # C. Attention Locale (Rien ne change ici)
        for layer in self.layers:
            x = layer(x, sorted_neighbors)
            
        x_local = self.local_agg(x) # [N, Hidden]
        
        # 3. --- Branche Globale ---
        x_global = self.global_encoder(data.pe_lap, data.pe_rw, data.pe_wl) # [N, Hidden]
        
        # 4. --- Noeud Virtuel (Contexte) ---
        graph_context = global_mean_pool(x_local, data.batch)
        x_virtual = graph_context[data.batch]
        
        # 5. --- Fusion ---
        x_combined = torch.cat([x_local, x_global, x_virtual], dim=-1)
        x_graph_level = global_mean_pool(x_combined, data.batch)
        
        # 6. --- Sortie ---
        z = self.final_mlp(x_graph_level)
        
        return z