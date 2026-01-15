import torch
import pickle
import numpy as np
import networkx as nx
import hashlib
import scipy.sparse as sp
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj, to_networkx
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_PATHS = ["data/train_graphs.pkl", "data/validation_graphs.pkl", "data/test_graphs.pkl"]
OUTPUT_PATHS = ["data/train_graphs_preprocessed.pkl", "data/validation_graphs_preprocessed.pkl", "data/test_graphs_preprocessed.pkl"]

MAX_K = 11
LPE_K = 8
RWSE_K = 16
WL_STEPS = 4

# Poids Heuristiques
BOND_WEIGHTS = {1: 1.0, 12: 1.5, 2: 2.0, 3: 3.0}
BOND_PRIORITIES = {'TRIPLE': 5, 'DOUBLE': 4, 'AROMATIC': 3, 'SINGLE': 2, 'HYDROGEN': 1}

# Mappings
x_map = {
    'atomic_num': list(range(0, 119)), # 0 à 118
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL'
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER'
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO'
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS'
    ],
    'is_conjugated': [False, True],
}
# ================= FONCTIONS DE CALCUL =================

def compute_global_features(data):
    """Calcule Laplacien + RWSE + WL avec gestion des petites molécules."""
    num_nodes = data.num_nodes

    # 1. Laplacien (LPE)
    if not hasattr(data, 'pe_lap'):
        edge_index, edge_weight = get_laplacian(data.edge_index, normalization='sym', num_nodes=num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        
        # FIX: Gestion des petites molécules
        # Si la molécule est plus petite que le nombre de vecteurs demandés (+ marge),
        # on utilise le solveur Dense (numpy) qui est stable pour les petites matrices.
        # Sinon, on utilise le solveur Sparse (scipy) qui est rapide pour les grosses matrices.
        
        if num_nodes < LPE_K + 2:
            # Cas Dense (Petite molécule)
            try:
                # On convertit en dense (.toarray) et on utilise eigh classique
                vals, vecs = np.linalg.eigh(L.toarray())
            except:
                # Fallback ultime (matrice vide ou buguée)
                vals, vecs = np.zeros(LPE_K), np.zeros((num_nodes, LPE_K))
        else:
            # Cas Sparse (Grosse molécule)
            try:
                vals, vecs = sp.linalg.eigsh(L, k=LPE_K + 1, which='SM', tol=1e-2)
            except:
                vals, vecs = np.zeros(LPE_K), np.zeros((num_nodes, LPE_K))

        # Post-traitement commun
        eig_vecs = vecs[:, 1:] # On enlève le premier vecteur propre (trivial, val=0)
        
        # Padding si on a moins de vecteurs que LPE_K (cas des petites molécules)
        if eig_vecs.shape[1] < LPE_K:
            pad = np.zeros((num_nodes, LPE_K - eig_vecs.shape[1]))
            eig_vecs = np.concatenate([eig_vecs, pad], axis=1)
            
        # On coupe si on en a trop (cas dense)
        eig_vecs = eig_vecs[:, :LPE_K]
        
        data.pe_lap = torch.from_numpy(eig_vecs).float()

    # 2. Random Walk (RWSE)
    if not hasattr(data, 'pe_rw'):
        try:
            A = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)[0]
            D = A.sum(dim=1)
            D_inv = torch.diag(1.0 / (D + 1e-6))
            P = D_inv @ A
            diagonals = []
            P_k = P.clone()
            for _ in range(RWSE_K):
                diagonals.append(P_k.diagonal())
                P_k = P_k @ P
            data.pe_rw = torch.stack(diagonals, dim=1)
        except:
             # Fallback RWSE
             data.pe_rw = torch.zeros((num_nodes, RWSE_K))

    # 3. WL (Weisfeiler-Lehman)
    if not hasattr(data, 'pe_wl'):
        row, col = data.edge_index
        current_colors = [str(int(data.x[i, 0].item())) for i in range(num_nodes)]
        wl_codes = []
        vocab_size = 50000
        wl_codes.append([int(hashlib.md5(c.encode()).hexdigest(), 16) % vocab_size for c in current_colors])
        
        for _ in range(WL_STEPS):
            next_colors = []
            for node in range(num_nodes):
                mask = (row == node)
                neighbors = col[mask]
                n_colors = sorted([current_colors[n.item()] for n in neighbors])
                sig = current_colors[node] + "_" + "_".join(n_colors)
                next_colors.append(hashlib.md5(sig.encode()).hexdigest())
            current_colors = next_colors
            wl_codes.append([int(c, 16) % vocab_size for c in current_colors])
        data.pe_wl = torch.tensor(wl_codes, dtype=torch.long).t().contiguous()

    return data

def get_sorted_neighbors(data):
    """
    Trie les voisins et renvoie deux tenseurs :
    1. Les indices des noeuds voisins (sorted_idx)
    2. Les indices des types de liaison (sorted_edge_idx)
    """
    num_nodes = data.num_nodes
    # On initialise avec -1 (padding)
    sorted_idx = torch.full((num_nodes, MAX_K), -1, dtype=torch.long)
    sorted_edge_idx = torch.full((num_nodes, MAX_K), -1, dtype=torch.long) # NOUVEAU

    row, col = data.edge_index
    
    for node_idx in range(num_nodes):
        mask = (row == node_idx)
        if not mask.any(): continue
        
        n_indices = col[mask]
        e_indices = torch.where(mask)[0]
        
        neighbors = []
        for i, n_idx in enumerate(n_indices):
            e_attr = data.edge_attr[e_indices[i]]
            n_feat = data.x[n_idx]
            
            # Récupération du type de liaison
            bond_idx = int(e_attr[0].item())
            
            # Score pour le tri (inchangé)
            bond_label = e_map['bond_type'][bond_idx] if bond_idx < len(e_map['bond_type']) else 'UNSPECIFIED'
            score_bond = BOND_PRIORITIES.get(bond_label, 0)
            
            z = x_map['atomic_num'][int(n_feat[0].item())]
            charge = abs(x_map['formal_charge'][int(n_feat[3].item())])
            in_ring = int(n_feat[8].item())
            
            # --- MODIFICATION ICI ---
            # On stocke aussi bond_idx dans le tuple (à la fin) pour ne pas le perdre
            neighbors.append(((score_bond, z, charge, in_ring), n_idx.item(), bond_idx))
            
        # Tri basé sur le premier élément du tuple (le score)
        neighbors.sort(key=lambda x: x[0], reverse=True)
        
        # On coupe à MAX_K
        top_k_neighbors = neighbors[:MAX_K]
        
        # Extraction séparée
        top_k_indices = [x[1] for x in top_k_neighbors] # Les noeuds
        top_k_bonds = [x[2] for x in top_k_neighbors]   # Les types de liaison
        
        # Remplissage des tenseurs
        len_k = len(top_k_neighbors)
        sorted_idx[node_idx, :len_k] = torch.tensor(top_k_indices)
        sorted_edge_idx[node_idx, :len_k] = torch.tensor(top_k_bonds) # NOUVEAU
        
    return sorted_idx, sorted_edge_idx

def compute_full_geometry(graph):
    """Génère la 3D Heuristique et calcule les features sans try/except."""
    
    # A. Génération 3D (Spring Layout)
    G = to_networkx(graph, to_undirected=True, remove_self_loops=True)
    rows, cols = graph.edge_index
    bond_types = graph.edge_attr[:, 0].long().tolist()
    
    for i in range(len(rows)):
        u, v = rows[i].item(), cols[i].item()
        if G.has_edge(u, v):
            G[u][v]['weight'] = BOND_WEIGHTS.get(bond_types[i], 1.0)

    # Si le graphe est vide ou a 0 noeud, networkx va lever une erreur ici.
    # C'est ce que tu veux : voir l'erreur.
    pos_dict = nx.spring_layout(G, dim=3, seed=42, iterations=200, threshold=1e-4, weight='weight')
    
    coords = np.array([pos_dict[i] for i in range(graph.num_nodes)])
    coords = coords - coords.mean(axis=0)
    graph.pos = torch.tensor(coords, dtype=torch.float32)

    # B. Calcul Angles/Distances
    sorted_neighbors, sorted_edges = get_sorted_neighbors(graph) 
    
    graph.sorted_neighbors = sorted_neighbors
    graph.sorted_edges = sorted_edges
    
    num_nodes = graph.num_nodes
    pos = graph.pos
    mask = (sorted_neighbors != -1)
    safe_indices = sorted_neighbors.clone()
    safe_indices[~mask] = 0
    
    central_pos = pos.unsqueeze(1)
    flat_idx = safe_indices.view(-1)
    neighbor_pos = pos[flat_idx].view(num_nodes, MAX_K, 3)
    
    rel_pos = neighbor_pos - central_pos
    dist = torch.norm(rel_pos, dim=-1) + 1e-6
    phi = torch.atan2(rel_pos[:, :, 1], rel_pos[:, :, 0])
    
    # Clamp est nécessaire mathématiquement sinon acos renvoie NaN pour 1.0000001
    z_norm = torch.clamp(rel_pos[:, :, 2] / dist, -1.0, 1.0)
    theta = torch.acos(z_norm)
    
    # [N, K, 5]
    geo_vec = torch.stack([dist, torch.sin(phi), torch.cos(phi), torch.sin(theta), torch.cos(theta)], dim=-1)
    graph.geo_features = geo_vec.float() * mask.unsqueeze(-1)
    
    return graph

# ================= MAIN =================
if __name__ == "__main__":
    for input_path, output_path in zip(INPUT_PATHS, OUTPUT_PATHS):
        print(f"Chargement de {input_path}...")
        with open(input_path, 'rb') as f:
            graphs = pickle.load(f)
            
        processed_graphs = []
        
        for i, g in enumerate(tqdm(graphs)):
            g = compute_global_features(g)
            g = compute_full_geometry(g)
            processed_graphs.append(g)
            
        print(f"Sauvegarde dans {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(processed_graphs, f)