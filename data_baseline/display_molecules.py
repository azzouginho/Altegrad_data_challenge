import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import numpy as np

# Dictionnaire de couleurs standard (CPK) pour les atomes communs
ATOM_COLORS = {
    1: '#FFFFFF',  # H (Blanc)
    6: '#909090',  # C (Gris)
    7: '#3050F8',  # N (Bleu)
    8: '#FF0D0D',  # O (Rouge)
    9: '#90E050',  # F
    15: '#FF8000', # P (Orange)
    16: '#FFFF30', # S (Jaune)
    17: '#1FF01F', # Cl
    35: '#A62929', # Br
    53: '#940094', # I
}

# Dictionnaire pour convertir Numéro Atomique -> Symbole
ATOM_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

def get_atom_prop(atomic_num):
    """Récupère la couleur et le symbole, avec des valeurs par défaut."""
    color = ATOM_COLORS.get(atomic_num, '#FF00FF') # Magenta si inconnu
    symbol = ATOM_SYMBOLS.get(atomic_num, str(atomic_num))
    return color, symbol

def visualize_molecule(pyg_graph, title="Molécule", show_indices=False):
    """
    Visualise un graphe PyG en utilisant NetworkX.
    
    Args:
        pyg_graph: L'objet Data de torch_geometric
        title: Titre du plot (ex: ID ou description)
        show_indices: Si True, affiche l'index du noeud au lieu de l'atome
    """
    plt.figure(figsize=(8, 8))
    
    # 1. Conversion PyG -> NetworkX
    # to_undirected=True car les graphes moléculaires sont physiques (non dirigés visuellement)
    G = to_networkx(pyg_graph, to_undirected=True)
    
    # 2. Récupération des numéros atomiques
    # On suppose que la feature 0 est le numéro atomique (standard dans data_utils.py)
    # x[:, 0] correspond à 'atomic_num' dans ton x_map
    atomic_nums = pyg_graph.x[:, 0].tolist()
    
    # 3. Préparation des labels et couleurs
    node_colors = []
    labels = {}
    
    for i, atom_num in enumerate(atomic_nums):
        atom_num = int(atom_num)
        color, symbol = get_atom_prop(atom_num)
        node_colors.append(color)
        
        if show_indices:
            labels[i] = f"{symbol}\n({i})"
        else:
            labels[i] = symbol

    # 4. Calcul du Layout (Positionnement)
    # kamada_kawai est souvent le plus "chimique" pour NetworkX sans coords 3D
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G)

    # 5. Dessin
    # Dessiner les noeuds
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors='black')
    
    # Dessiner les arêtes (liaisons)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7)
    
    # Dessiner les labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight='bold')

    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    from data_utils import PreprocessedGraphDataset
    import os

    GRAPH_PATH = "data/test_graphs_preprocessed.pkl" 
    
    if os.path.exists(GRAPH_PATH):
        # Chargement du dataset
        dataset = PreprocessedGraphDataset(graph_path=GRAPH_PATH)
        
        # On prend un exemple (le premier du dataset)
        sample_graph = dataset[0][0]
        print(sample_graph)
        
        # Récupération de la description si disponible dans l'objet graph
        desc = getattr(sample_graph, 'description', f"ID: {sample_graph.id}")
        
        print(f"Visualisation de la molécule {sample_graph.id}")
        print(f"Nombre d'atomes: {sample_graph.num_nodes}")
        
        # Visualisation
        visualize_molecule(sample_graph, title=desc[:50] + "...", show_indices=True) # On coupe le titre si trop long
    else:
        print(f"Fichier {GRAPH_PATH} introuvable pour la démo.")