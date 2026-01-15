import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn
from gat_encoder import MolGATEncoder


def contrastive_loss(g: torch.Tensor, t: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    CLIP / InfoNCE style:
      g, t: [B, D] normalized
      logits = g @ t.T / tau
      CE in both directions
    """
    logits = (g @ t.t()) / tau
    targets = torch.arange(g.size(0), device=g.device)
    loss_gt = F.cross_entropy(logits, targets)
    loss_tg = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_gt + loss_tg)


def train_epoch(model, loader, optim, device, loss_type: str, tau: float):
    model.train()
    total, total_loss = 0, 0.0

    for graphs, text_emb in tqdm(loader):
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        txt = F.normalize(text_emb, dim=-1)

        g, _ = model(graphs)  # g: [B, D]
        if loss_type == "mse":
            loss = F.mse_loss(g, txt)
        else:
            loss = contrastive_loss(g, txt, tau=tau)

        optim.zero_grad()
        loss.backward()
        optim.step()

        bs = graphs.num_graphs
        total += bs
        total_loss += loss.item() * bs

    return total_loss / max(total, 1)


@torch.no_grad()
def eval_retrieval(model, loader, device):
    """
    Simple retrieval metric: MRR on the validation split using in-split matching.
    """
    model.eval()
    all_g, all_t = [], []
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        g, _ = model(graphs)
        all_g.append(g)
        all_t.append(F.normalize(text_emb, dim=-1))
    all_g = torch.cat(all_g, dim=0)
    all_t = torch.cat(all_t, dim=0)

    sims = all_t @ all_g.t()
    ranks = sims.argsort(dim=-1, descending=True)
    n = all_t.size(0)
    correct = torch.arange(n, device=device)
    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1
    mrr = (1.0 / pos.float()).mean().item()
    return {"MRR": mrr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_graphs", default="data/train_graphs.pkl")
    ap.add_argument("--val_graphs", default="data/validation_graphs.pkl")
    ap.add_argument("--train_emb", default="data/train_embeddings.csv")
    ap.add_argument("--val_emb", default="data/validation_embeddings.csv")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loss", choices=["mse", "contrastive"], default="contrastive")
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--save", default="gat_align.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_emb = load_id2emb(args.train_emb)
    val_emb = load_id2emb(args.val_emb) if os.path.exists(args.val_emb) else None
    out_dim = len(next(iter(train_emb.values())))

    train_ds = PreprocessedGraphDataset(args.train_graphs, train_emb)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    val_dl = None
    if val_emb is not None and os.path.exists(args.val_graphs):
        val_ds = PreprocessedGraphDataset(args.val_graphs, val_emb)
        val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = MolGATEncoder(
        hidden_dim=args.hidden,
        num_layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        out_dim=out_dim,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_mrr = -1.0
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_dl, optim, device, args.loss, args.tau)
        if val_dl is not None:
            scores = eval_retrieval(model, val_dl, device)
            mrr = scores["MRR"]
        else:
            scores = {}
            mrr = -1.0
        print(f"Epoch {ep}/{args.epochs} | loss={tr_loss:.4f} | val={scores}")

        if mrr > best_mrr:
            best_mrr = mrr
            torch.save(model.state_dict(), args.save)

    print("Saved best model to", args.save)


if __name__ == "__main__":
    main()
