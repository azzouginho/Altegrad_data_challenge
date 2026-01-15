import torch
import pickle
import numpy as np
from tqdm import tqdm

# Configuration
DATA_PATH = "data/train_graphs_preprocessed.pkl"

# Les dimensions définies dans ton layers.py
ATOM_FEATURE_DIMS = [119, 10, 11, 12, 9, 5, 8, 2, 2]

# Les noms pour s'y retrouver
FEATURE_NAMES = [
    "0. Atomic Num      ",
    "1. Chirality       ",
    "2. Degree          ",
    "3. Formal Charge   ",
    "4. Num Hs          ",
    "5. Radical Electr  ",
    "6. Hybridization   ",
    "7. Is Aromatic     ",
    "8. Is In Ring      "
]

def inspect():
    print(f"Chargement de {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        graphs = pickle.load(f)
    
    print(f"Analyse de {len(graphs)} graphes...")
    
    # Stockage des min/max observés
    # On initialise avec des valeurs inverses
    min_vals = [9999] * 9
    max_vals = [-9999] * 9
    
    # Stockage des valeurs uniques pour comprendre la distribution
    unique_vals = [set() for _ in range(9)]
    
    for g in tqdm(graphs):
        x = g.x # [Num_Nodes, 9]
        
        for i in range(9):
            col = x[:, i]
            
            # Mise à jour Min/Max
            curr_min = col.min().item()
            curr_max = col.max().item()
            
            if curr_min < min_vals[i]: min_vals[i] = curr_min
            if curr_max > max_vals[i]: max_vals[i] = curr_max
            
            # On ajoute les valeurs uniques (attention si c'est gros, mais ici c'est des entiers)
            # On convertit en liste pour update (plus rapide par batch)
            unique_vals[i].update(col.tolist())

    print("\n" + "="*80)
    print(f"{'FEATURE':<20} | {'MIN':<5} | {'MAX':<5} | {'LIMIT (DIM)':<12} | {'STATUS'}")
    print("="*80)
    
    for i in range(9):
        name = FEATURE_NAMES[i]
        obs_min = min_vals[i]
        obs_max = max_vals[i]
        limit = ATOM_FEATURE_DIMS[i]
        
        # Pour la charge, on simule l'offset fait dans layers.py (+5)
        # Si i == 3, l'index réel utilisé par l'embedding sera val + 5
        if i == 1999:
            check_max = obs_max + 5
            check_min = obs_min + 5
            info_supp = f"(Raw: {obs_min} to {obs_max})"
        else:
            check_max = obs_max
            check_min = obs_min
            info_supp = ""
            
        # Vérification du crash
        # L'index doit être strictement inférieur à la limite (0 à Limit-1)
        if check_max >= limit or check_min < 0:
            status = "❌ CRASH"
        else:
            status = "✅ OK"
            
        print(f"{name} | {check_min:<5} | {check_max:<5} | {limit:<12} | {status} {info_supp}")

    print("="*80)
    print("NOTE : Pour 'Formal Charge', les valeurs affichées MIN/MAX tiennent compte de l'offset +5 du modèle.")
    print("Si STATUS est CRASH, c'est que tes données contiennent des valeurs non prévues par tes Embeddings.")

if __name__ == "__main__":
    inspect()