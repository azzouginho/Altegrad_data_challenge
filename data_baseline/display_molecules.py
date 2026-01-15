import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import numpy as np

ATOM_COLORS = {
    1: '#FFFFFF',  # H
    6: '#909090',  # C
    7: '#3050F8',  # N
    8: '#FF0D0D',  # O
    9: '#90E050',  # F
    15: '#FF8000', # P
    16: '#FFFF30', # S
    17: '#1FF01F', # Cl
    35: '#A62929', # Br
    53: '#940094', # I
}

ATOM_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

def get_atom_prop(atomic_num):
    """Get the color and symbol of an atom"""
    color = ATOM_COLORS.get(atomic_num, '#FF00FF')
    symbol = ATOM_SYMBOLS.get(atomic_num, str(atomic_num))
    return color, symbol

def visualize_molecule(pyg_graph, title="Molécule", show_indices=False):
    """
    Visualization of the graph.
    
    Args:
        pyg_graph: Data object of torch_geometric
        title: Title (ex: ID or description)
        show_indices: If True, displays the index of the node instead of the atom
    """
    plt.figure(figsize=(8, 8))
    
    G = to_networkx(pyg_graph, to_undirected=True)
    atomic_nums = pyg_graph.x[:, 0].tolist()
    
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

    # Positions of the nodes
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight='bold')

    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    from data_utils import PreprocessedGraphDataset
    import os

    GRAPH_PATH = "data/test_graphs_preprocessed.pkl" 
    
    if os.path.exists(GRAPH_PATH):
        dataset = PreprocessedGraphDataset(graph_path=GRAPH_PATH)
        
        # We take an example
        sample_graph = dataset[0][0]
        
        # We get the caption if available
        desc = getattr(sample_graph, 'description', f"ID: {sample_graph.id}")
        
        print(f"Visualization of the molecule {sample_graph.id}")
        print(f"Number of atoms : {sample_graph.num_nodes}")
        
        visualize_molecule(sample_graph, title=desc[:50] + "...", show_indices=True) # Cut the title if too long
    else:
        print(f"File {GRAPH_PATH} not found.")