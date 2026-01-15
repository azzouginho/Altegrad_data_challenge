"""
data_utils.py
Version Optimisée : Charge simplement les données pré-calculées.
Aucun calcul lourd ici (tout est déjà dans le .pkl généré par preprocessing.py).
"""
import torch
import pickle
import pandas as pd
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from torch_geometric.data import Batch

# =========================================================
# 1. Dataset Ultra-Léger
# =========================================================
class PreprocessedGraphDataset(Dataset):
    def __init__(self, graph_path: str, emb_dict: Dict[str, torch.Tensor] = None):
        """
        Args:
            graph_path: Chemin vers le fichier .pkl FINAL (généré par preprocessing.py)
            emb_dict: Dictionnaire {id: embedding_texte} pour le training
        """
        print(f"📂 Loading pre-computed graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
            
        self.emb_dict = emb_dict
        print(f"✅ Loaded {len(self.graphs)} graphs ready for training.")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # 1. Récupération du graphe (Lecture mémoire directe)
        graph = self.graphs[idx]
        
        # 2. Extraction des features pré-calculées
        # Ces attributs existent déjà car créés par preprocessing.py
        # sorted_neighbors : [N, K] (Indices des voisins triés)
        # geo_features     : [N, K, 5] (Dist, Sin/Cos Phi, Sin/Cos Theta)
        
        # Note: pe_lap, pe_rw, pe_wl sont stockés directement dans l'objet 'graph' 
        # (graph.pe_lap, etc.) et seront gérés automatiquement par PyG.

        if self.emb_dict is not None:
            # Gestion ID (parfois int, parfois str selon les CSV)
            g_id = str(graph.id) 
            text_emb = self.emb_dict[g_id]
            
            # On renvoie tout ce dont le modèle a besoin
            return graph, graph.sorted_neighbors, graph.geo_features, text_emb
        else:
            # Mode Inférence (pas de texte)
            return graph, graph.sorted_neighbors, graph.geo_features

# =========================================================
# 2. Collation (Batching)
# =========================================================
def collate_fn(batch):
    """
    Assemble un batch de graphes et de features.
    Gère le décalage (offset) des indices des voisins et la concaténation des features 3D.
    """
    # Détection si on a le texte ou pas
    has_text = len(batch[0]) == 4
    
    if has_text:
        graphs, neighbors_list, geo_list, text_embs = zip(*batch)
        text_embs = torch.stack(text_embs, dim=0) # [Batch, Text_Dim]
    else:
        graphs, neighbors_list, geo_list = zip(*batch)
        text_embs = None

    # A. Batching PyG Standard
    # Batch.from_data_list empile automatiquement x, edge_index, pos, pe_lap, pe_rw, etc.
    batch_graph = Batch.from_data_list(list(graphs))
    
    # B. Batching des Voisins (Indices avec Offset)
    # Problème : Le voisin '0' du 2ème graphe doit devenir 'N_graphe_1' dans le batch global.
    adjusted_neighbors = []
    offset = 0
    for i, local_neighbors in enumerate(neighbors_list):
        # On ne touche pas au padding (-1)
        mask = (local_neighbors != -1)
        
        global_neighbors = local_neighbors.clone()
        global_neighbors[mask] += offset # Ajout de l'offset
        
        adjusted_neighbors.append(global_neighbors)
        
        # Mise à jour de l'offset pour le prochain graphe
        offset += graphs[i].num_nodes
        
    total_sorted_neighbors = torch.cat(adjusted_neighbors, dim=0) # [Total_Nodes, K]
    
    # C. Batching des Features Géométriques 3D
    # Pas d'offset à gérer, juste une concaténation car ce sont des floats [Dist, Angles...]
    total_geo_features = torch.cat(geo_list, dim=0) # [Total_Nodes, K, 5]
    
    if has_text:
        return batch_graph, total_sorted_neighbors, total_geo_features, text_embs
    else:
        return batch_graph, total_sorted_neighbors, total_geo_features

# =========================================================
# 3. Helpers Chargement Texte
# =========================================================
def load_id2emb(csv_path: str) -> Dict[str, torch.Tensor]:
    """Charge les embeddings textuels depuis le CSV."""
    print(f"📖 Loading text embeddings from {csv_path}...")
    df = pd.read_csv(csv_path)
    id2emb = {}
    for _, row in df.iterrows():
        # Parsing de la string "[0.1, 0.5, ...]" si nécessaire
        try:
            if isinstance(row["embedding"], str):
                 vals = [float(x) for x in row["embedding"].replace('[', '').replace(']', '').split(',')]
            else:
                 vals = row["embedding"] # Si déjà liste
            id2emb[str(row["ID"])] = torch.tensor(vals, dtype=torch.float32)
        except Exception as e:
            print(f"Error parsing embedding for ID {row['ID']}: {e}")
            continue
            
    return id2emb