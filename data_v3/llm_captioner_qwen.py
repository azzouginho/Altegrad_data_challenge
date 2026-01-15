import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


class GraphPrefixQwenCaptioner(nn.Module):
    """
    Graph -> node embeddings (from a graph encoder, e.g. GAT) -> project to Qwen hidden size -> prepend as prefix tokens.

    IMPORTANT:
      - <graph> is only a TEXT marker inside the prompt (a cue for the LLM).
      - The actual graph information is in the prefix embeddings.
    """

    def __init__(
        self,
        graph_encoder,
        llm_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        freeze_graph: bool = True,
        max_nodes: int = 128,
        load_in_4bit: bool = True,
        # LoRA
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        # T4 typically prefers float16 compute; A100/L4 prefer bfloat16
        bnb_compute_dtype: str = "float16",  # "float16" or "bfloat16"
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.freeze_graph = freeze_graph
        self.max_nodes = max_nodes

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add a marker token (prompt-only)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<graph>"]})

        quant_cfg = None
        if load_in_4bit:
            compute_dtype = torch.float16 if bnb_compute_dtype == "float16" else torch.bfloat16
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        # For Colab GPUs, float16 is usually the safest default
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=quant_cfg,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Resize embeddings since we added a token
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # LoRA
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(target_modules),
        )
        self.llm = get_peft_model(self.llm, lora_cfg)

        # Projection from graph node dim -> LLM hidden dim
        self.llm_dim = self.llm.config.hidden_size
        graph_dim = getattr(graph_encoder, "hidden_dim", None)
        if graph_dim is None:
            raise ValueError("graph_encoder must expose hidden_dim (node embedding size).")

        self.project = nn.Sequential(
            nn.LayerNorm(graph_dim),
            nn.Linear(graph_dim, self.llm_dim),
        )

    def _model_embed_dtype(self):
        return self.llm.get_input_embeddings().weight.dtype

    def _encode_graph_prefix(self, graphs):
        """
        Returns:
          prefix_embeds: [B, V, llm_dim]
          prefix_mask:   [B, V] long  (1 real node, 0 pad)
        """
        if self.freeze_graph:
            self.graph_encoder.eval()
            with torch.no_grad():
                _, h_node = self.graph_encoder(graphs)  # [N, graph_dim]
        else:
            _, h_node = self.graph_encoder(graphs)

        dense, mask = to_dense_batch(h_node, graphs.batch)  # [B,V,D], [B,V] bool
        if dense.size(1) > self.max_nodes:
            dense = dense[:, : self.max_nodes, :]
            mask = mask[:, : self.max_nodes]

        prefix = self.project(dense)  # [B,V,llm_dim] float32 by default
        prefix_mask = mask.to(dtype=torch.long)

        # Cast prefix to the model's embedding dtype (usually float16)
        prefix = prefix.to(dtype=self._model_embed_dtype())
        return prefix, prefix_mask

    def build_chat(self, user_text: str, assistant_text: str | None = None):
        messages = [
            {"role": "system", "content": "You are an expert chemist."},
            {"role": "user", "content": user_text},
        ]
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def tokenize_prompt(self, messages, add_generation_prompt: bool):
        """
        Uses Qwen chat template so formatting is correct.
        Returns 1D tensors: input_ids, attention_mask
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        enc = self.tokenizer(text, return_tensors="pt", padding=False, truncation=True)
        return enc["input_ids"][0], enc["attention_mask"][0]

    def forward(self, graphs, input_ids, attention_mask, labels):
        """
        Training forward:
          - embed text tokens
          - prepend graph prefix embeddings
          - pass ONLY inputs_embeds to Qwen (NOT input_ids)
          - ensure dtype matches model weights
        """
        device = next(self.llm.parameters()).device
        model_dtype = self._model_embed_dtype()

        graphs = graphs.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        prefix, prefix_mask = self._encode_graph_prefix(graphs)
        prefix = prefix.to(device=device, dtype=model_dtype)
        prefix_mask = prefix_mask.to(device)

        B, V, _ = prefix.shape

        # Text embeddings (cast to model dtype)
        text_embeds = self.llm.get_input_embeddings()(input_ids).to(dtype=model_dtype)

        # Concatenate
        inputs_embeds = torch.cat([prefix, text_embeds], dim=1)            # [B,V+T,dim]
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)        # [B,V+T]

        # Ignore loss on prefix positions
        ignore = torch.full((B, V), -100, dtype=labels.dtype, device=device)
        full_labels = torch.cat([ignore, labels], dim=1)                   # [B,V+T]

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )
        return out

    @torch.no_grad()
    def generate_batch(self, graphs, user_text: str, max_new_tokens: int = 128, temperature: float = 0.8):
        """
        Batched generation. Returns list[str] of size batch.
        """
        device = next(self.llm.parameters()).device
        model_dtype = self._model_embed_dtype()

        graphs = graphs.to(device)

        prefix, prefix_mask = self._encode_graph_prefix(graphs)
        prefix = prefix.to(device=device, dtype=model_dtype)
        prefix_mask = prefix_mask.to(device)

        B, V, _ = prefix.shape

        # Prompt ids (chat template)
        msgs = self.build_chat(user_text=user_text, assistant_text=None)
        prompt_ids_1d, prompt_mask_1d = self.tokenize_prompt(msgs, add_generation_prompt=True)
        prompt_ids = prompt_ids_1d.unsqueeze(0).repeat(B, 1).to(device)
        prompt_mask = prompt_mask_1d.unsqueeze(0).repeat(B, 1).to(device)

        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids).to(dtype=model_dtype)

        inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)          # [B,V+T,dim]
        full_mask = torch.cat([prefix_mask, prompt_mask], dim=1)           # [B,V+T]

        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # Heuristic stripping: remove prompt if echoed
        outs = []
        for s in decoded:
            s2 = s.strip()
            if user_text in s2:
                s2 = s2.split(user_text, 1)[-1].strip()
            outs.append(s2)
        return outs