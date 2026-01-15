import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data_utils import PreprocessedGraphDataset, collate_fn, load_id2emb
from model import FullEncoderModel

# =========================================================
# 1. Configuration
# =========================================================
CONFIG = {
    'graph_path': 'data/train_graphs_preprocessed.pkl', 
    'emb_path': 'data/train_embeddings.csv',
    'save_dir': 'checkpoints',
    
    'lap_dim': 8,
    'rw_dim': 16,
    'wl_vocab': 50000,
    'hidden_dim': 256,
    'output_dim': 768,
    
    'batch_size': 128,
    'lr': 1e-4,
    'epochs': 50,
    'temperature': 0.07, 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# =========================================================
# 2. La Loss Contrastive (InfoNCE)
# =========================================================
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, graph_emb, text_emb):
        # 1. Normalisation
        graph_emb = nn.functional.normalize(graph_emb, dim=-1)
        text_emb = nn.functional.normalize(text_emb, dim=-1)
        
        # 2. Similarité
        logits = torch.matmul(graph_emb, text_emb.T) / self.temperature
        
        # 3. Targets
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        # 4. Loss Symétrique
        loss_g2t = self.cross_entropy(logits, labels)
        loss_t2g = self.cross_entropy(logits.T, labels)
        
        return (loss_g2t + loss_t2g) / 2

# =========================================================
# 3. Boucle d'Entraînement
# =========================================================
def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']
    print(f"Démarrage sur {device}")

    # --- A. Data ---
    print("Chargement des données...")
    emb_dict = load_id2emb(CONFIG['emb_path'])
    
    # Dataset simplifié : plus besoin de passer les params structurels (déjà dans le .pkl)
    dataset = PreprocessedGraphDataset(
        CONFIG['graph_path'], 
        emb_dict=emb_dict
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    # --- B. Modèle ---
    model = FullEncoderModel(
        lap_dim=CONFIG['lap_dim'],
        rw_dim=CONFIG['rw_dim'],
        wl_vocab_size=CONFIG['wl_vocab'],
        hidden_dim=CONFIG['hidden_dim'],
        output_dim=CONFIG['output_dim']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
    criterion = ContrastiveLoss(temperature=CONFIG['temperature'])

    # --- C. Loop ---
    model.train()
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in progress_bar:
            # Graph, Voisins, Géo, Texte
            graph_data, sorted_neighbors, geo_features, text_targets = batch
            
            # Move to GPU
            graph_data = graph_data.to(device)
            sorted_neighbors = sorted_neighbors.to(device)
            geo_features = geo_features.to(device) # <-- Ne pas oublier
            text_targets = text_targets.to(device)
            
            optimizer.zero_grad()
            
            # --- Forward avec Géométrie 3D ---
            z_graph = model(graph_data, sorted_neighbors, geo_features) 
            
            # Loss Contrastive
            loss = criterion(z_graph, text_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Terminée | Loss Moyenne: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), f"{CONFIG['save_dir']}/model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()