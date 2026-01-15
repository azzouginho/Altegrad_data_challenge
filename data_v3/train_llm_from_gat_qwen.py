import os
import argparse
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from tqdm import tqdm

from gat_encoder import MolGATEncoder
from llm_captioner_qwen import GraphPrefixQwenCaptioner


class MolCaptionDataset(Dataset):
    def __init__(self, graph_pkl: str):
        with open(graph_pkl, "rb") as f:
            self.graphs = pickle.load(f)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        return g, g.description


def collate_caption(batch, captioner: GraphPrefixQwenCaptioner, max_len: int = 256):
    graphs, descs = zip(*batch)
    g_batch = Batch.from_data_list(list(graphs))

    # Chat format: user includes the <graph> marker
    user_texts = ["<graph>\nDescribe this molecule." for _ in descs]

    # We need prompt_len per example to mask labels correctly.
    # We'll build full messages including assistant answer, tokenize, and mask prompt tokens.
    input_ids_list, attn_list, labels_list = [], [], []
    for u, ans in zip(user_texts, descs):
        # Prompt-only (with generation prompt)
        prompt_msgs = captioner.build_chat(user_text=u, assistant_text=None)
        prompt_ids, _ = captioner.tokenize_prompt(prompt_msgs, add_generation_prompt=True)

        # Full conversation including assistant answer
        full_msgs = captioner.build_chat(user_text=u, assistant_text=ans)
        full_text = captioner.tokenizer.apply_chat_template(
            full_msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = captioner.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]

        labels = ids.clone()
        prompt_len = prompt_ids.numel()
        labels[:prompt_len] = -100
        labels[attn == 0] = -100

        input_ids_list.append(ids)
        attn_list.append(attn)
        labels_list.append(labels)

    # Pad to batch max
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=captioner.tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

    return g_batch, input_ids, attention_mask, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_graphs", default="data/train_graphs.pkl")
    ap.add_argument("--gat_ckpt", default="gat_align.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--save_dir", default="qwen05b_lora_captioner")
    ap.add_argument("--max_nodes", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Build GAT encoder (node dim hidden_dim). out_dim not used for captioning.
    graph_encoder = MolGATEncoder(out_dim=768)
    graph_encoder.load_state_dict(torch.load(args.gat_ckpt, map_location="cpu"))
    graph_encoder.to(device)

    print(1)

    captioner = GraphPrefixQwenCaptioner(
        graph_encoder=graph_encoder,
        llm_name="Qwen/Qwen2.5-0.5B-Instruct",
        freeze_graph=True,
        max_nodes=args.max_nodes,
        load_in_4bit=True,
        # LoRA defaults are good for Qwen
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    print(2)
    captioner.to(device)
    print(3)
    ds = MolCaptionDataset(args.train_graphs)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_caption(b, captioner, max_len=args.max_len),
    )
    print(4)
    # Only LoRA params require grad
    optim = torch.optim.AdamW(captioner.parameters(), lr=args.lr)
    print(5)
    captioner.train()
    for ep in range(1, args.epochs + 1):
        total_loss, total = 0.0, 0
        for graphs, input_ids, attn, labels in tqdm(dl):
            graphs = graphs.to(next(captioner.llm.parameters()).device)
            input_ids = input_ids.to(next(captioner.llm.parameters()).device)
            attn = attn.to(next(captioner.llm.parameters()).device)
            labels = labels.to(next(captioner.llm.parameters()).device)

            out = captioner(graphs, input_ids, attn, labels)
            loss = out.loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            bs = graphs.num_graphs
            total += bs
            total_loss += loss.item() * bs

        print(f"Epoch {ep}/{args.epochs} | loss={total_loss/max(total,1):.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    captioner.llm.save_pretrained(args.save_dir)
    captioner.tokenizer.save_pretrained(args.save_dir)
    print("Saved LoRA adapters + tokenizer to", args.save_dir)


if __name__ == "__main__":
    main()
