import torch
import pickle
import numpy as np
import networkx as nx
import hashlib
import scipy.sparse as sp
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj, to_networkx
from tqdm import tqdm

INPUT_PATHS = ["data/train_graphs.pkl", "data/validation_graphs.pkl", "data/test_graphs.pkl"]
OUTPUT_PATHS = ["data/train_graphs_preprocessed.pkl", "data/validation_graphs_preprocessed.pkl", "data/test_graphs_preprocessed.pkl"]

MAX_K = 11 # Max degree

# Parameters for graph features
LPE_K = 8
RWSE_K = 16
WL_STEPS = 4

# Weights for neighbors sorting
BOND_WEIGHTS = {1: 1.0, 12: 1.5, 2: 2.0, 3: 3.0}
BOND_PRIORITIES = {'TRIPLE': 5, 'DOUBLE': 4, 'AROMATIC': 3, 'SINGLE': 2, 'HYDROGEN': 1}

x_map = {
    'atomic_num': list(range(0, 119)),
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

def compute_global_features(data):
    """Computes Laplacian + RWSE + WL."""
    num_nodes = data.num_nodes

    # 1. Laplacian (LPE)
    if not hasattr(data, 'pe_lap'):
        edge_index, edge_weight = get_laplacian(data.edge_index, normalization='sym', num_nodes=num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        
        # For small graphs
        if num_nodes < LPE_K + 2:
            try:
                vals, vecs = np.linalg.eigh(L.toarray())
            except:
                vals, vecs = np.zeros(LPE_K), np.zeros((num_nodes, LPE_K))
        # General case for large graphs
        else:
            try:
                vals, vecs = sp.linalg.eigsh(L, k=LPE_K + 1, which='SM', tol=1e-2)
            except:
                vals, vecs = np.zeros(LPE_K), np.zeros((num_nodes, LPE_K))

        eig_vecs = vecs[:, 1:] # First one is trivial
        
        # Padding if needed
        if eig_vecs.shape[1] < LPE_K:
            pad = np.zeros((num_nodes, LPE_K - eig_vecs.shape[1]))
            eig_vecs = np.concatenate([eig_vecs, pad], axis=1)
            
        # We cut if too much dense
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
    """Canonical sorting of neighbors using the deterministic weights we defined"""
    num_nodes = data.num_nodes
    sorted_idx = torch.full((num_nodes, MAX_K), -1, dtype=torch.long)
    sorted_edge_idx = torch.full((num_nodes, MAX_K), -1, dtype=torch.long)

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
            
            bond_idx = int(e_attr[0].item())
            
            # Score to sort
            bond_label = e_map['bond_type'][bond_idx] if bond_idx < len(e_map['bond_type']) else 'UNSPECIFIED'
            score_bond = BOND_PRIORITIES.get(bond_label, 0)
            
            z = x_map['atomic_num'][int(n_feat[0].item())]
            charge = abs(x_map['formal_charge'][int(n_feat[3].item())])
            in_ring = int(n_feat[8].item())
            
            neighbors.append(((score_bond, z, charge, in_ring), n_idx.item(), bond_idx))
            
        # Based on the score
        neighbors.sort(key=lambda x: x[0], reverse=True)
        top_k_neighbors = neighbors[:MAX_K] # Not needed with clean data but just in case
        
        # We need both to use both node and edge features later
        top_k_indices = [x[1] for x in top_k_neighbors] # Nodes
        top_k_bonds = [x[2] for x in top_k_neighbors]   # Edges
        
        len_k = len(top_k_neighbors)
        sorted_idx[node_idx, :len_k] = torch.tensor(top_k_indices)
        sorted_edge_idx[node_idx, :len_k] = torch.tensor(top_k_bonds)
        
    return sorted_idx, sorted_edge_idx

def compute_full_geometry(graph):
    """Generate the 3D in a heuristic way and computes features."""
    
    # A. 3D Generation
    G = to_networkx(graph, to_undirected=True, remove_self_loops=True)
    rows, cols = graph.edge_index
    bond_types = graph.edge_attr[:, 0].long().tolist()
    
    for i in range(len(rows)):
        u, v = rows[i].item(), cols[i].item()
        if G.has_edge(u, v):
            G[u][v]['weight'] = BOND_WEIGHTS.get(bond_types[i], 1.0)

    pos_dict = nx.spring_layout(G, dim=3, seed=42, iterations=200, threshold=1e-4, weight='weight')
    
    coords = np.array([pos_dict[i] for i in range(graph.num_nodes)])
    coords = coords - coords.mean(axis=0)
    graph.pos = torch.tensor(coords, dtype=torch.float32)

    # B. Angles and distances
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
    
    # Clamp to prevent issues with numerical stability (ex : arcos(1.0001))
    z_norm = torch.clamp(rel_pos[:, :, 2] / dist, -1.0, 1.0)
    theta = torch.acos(z_norm)
    
    # [N, K, 5]
    geo_vec = torch.stack([dist, torch.sin(phi), torch.cos(phi), torch.sin(theta), torch.cos(theta)], dim=-1)
    graph.geo_features = geo_vec.float() * mask.unsqueeze(-1)
    
    return graph

if __name__ == "__main__":
    for input_path, output_path in zip(INPUT_PATHS, OUTPUT_PATHS):
        print(f"Loading {input_path}...")
        with open(input_path, 'rb') as f:
            graphs = pickle.load(f)
            
        processed_graphs = []
        
        for i, g in enumerate(tqdm(graphs)):
            g = compute_global_features(g)
            g = compute_full_geometry(g)
            processed_graphs.append(g)
            
        print(f"Saving into {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(processed_graphs, f)