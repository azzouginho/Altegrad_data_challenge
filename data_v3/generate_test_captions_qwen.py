import argparse
import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch.utils.data import Subset

from peft import PeftModel

from tqdm import tqdm

from gat_encoder import MolGATEncoder
from llm_captioner_qwen import GraphPrefixQwenCaptioner


class GraphOnlyDataset(Dataset):
    def __init__(self, graph_pkl: str):
        with open(graph_pkl, "rb") as f:
            self.graphs = pickle.load(f)
        self.ids = [g.id for g in self.graphs]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_graphs(batch):
    return Batch.from_data_list(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_path", default="data/test_graphs.pkl")
    ap.add_argument("--gat_ckpt", default="gat_align.pt")
    ap.add_argument("--lora_dir", default="qwen05b_lora_captioner")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_csv", default="submission.csv")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_nodes", type=int, default=128)
    ap.add_argument("--split", default='train')
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    graph_encoder = MolGATEncoder(out_dim=768)
    graph_encoder.load_state_dict(torch.load(args.gat_ckpt, map_location="cpu"))
    graph_encoder.to(device)

    captioner = GraphPrefixQwenCaptioner(
        graph_encoder=graph_encoder,
        llm_name="Qwen/Qwen2.5-0.5B-Instruct",
        freeze_graph=True,
        max_nodes=args.max_nodes,
        load_in_4bit=True,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    captioner.to(device)

    # ✅ Robust LoRA load across PEFT versions
    captioner.llm = PeftModel.from_pretrained(captioner.llm, args.lora_dir)
    captioner.llm.eval()

    ds = GraphOnlyDataset(args.graphs_path)
    graphs_list = ds.graphs
    ids = ds.ids
    if args.split == 'train':
        ds = Subset(ds, range(100))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    all_rows = []
    seen = 0

    for graphs in tqdm(dl):
        graphs = graphs.to(next(captioner.llm.parameters()).device)

        user_text = "<graph>\nDescribe this molecule."
        outs = captioner.generate_batch(
            graphs,
            user_text=user_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        if args.split == 'test':
            for i, txt in enumerate(outs):
                all_rows.append({"ID": ids[seen + i], "description": txt})
            seen += graphs.num_graphs

        elif args.split == 'train':
            for i, txt in enumerate(outs):
                all_rows.append({"ID": ids[seen + i], "description": txt,
                                 "ground_truth":graphs_list[seen + i].description})
            seen += graphs.num_graphs

        if seen % 200 == 0:
            print("Generated", seen)

    pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv)


if __name__ == "__main__":
    main()
