"""Federated NanoGPT on Shakespeare using flwr_serverless.

Simulates 2 federated nodes in a single process using AsyncFederatedNode
and InMemoryFolder. Each node trains a character-level GPT on a partition
of the Tiny Shakespeare dataset, then aggregates weights via FedAvg.

Usage:
    python examples/nanogpt_shakespeare.py
"""

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from flwr_serverless import AsyncFederatedNode
from flwr_serverless.shared_folder.in_memory_folder import InMemoryFolder

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _download_shakespeare() -> str:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "input.txt"
    if not path.exists():
        text = requests.get(DATA_URL).text
        path.write_text(text)
    return path.read_text()


def _get_meta():
    meta_path = DATA_DIR / "meta.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    text = _download_shakespeare()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    meta = {"vocab_size": len(chars), "stoi": stoi, "itos": itos}
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    return meta


def _encode(text: str, stoi: dict) -> list:
    return [stoi[c] for c in text]


class ShakespeareDataset(Dataset):
    def __init__(self, data: np.ndarray, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(1, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(
            self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64)
        )
        return x, y


def load_data(partition_id: int, num_partitions: int, batch_size: int, block_size: int):
    text = _download_shakespeare()
    meta = _get_meta()
    ids = np.array(_encode(text, meta["stoi"]), dtype=np.uint16)

    n = len(ids)
    split = int(n * 0.9)
    train_ids = ids[:split]
    val_ids = ids[split:]

    chunk_size = len(train_ids) // num_partitions
    start = partition_id * chunk_size
    end = start + chunk_size if partition_id < num_partitions - 1 else len(train_ids)
    partition_train = train_ids[start:end]

    train_ds = ShakespeareDataset(partition_train, block_size)
    val_ds = ShakespeareDataset(val_ids, block_size)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return trainloader, valloader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Train / Test
# ---------------------------------------------------------------------------


def train(model, trainloader, epochs, lr, device, max_steps=0):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    running_loss = 0.0
    n_batches = 0
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            if max_steps > 0 and n_batches >= max_steps:
                return running_loss / n_batches
    return running_loss / max(n_batches, 1)


def test(model, testloader, device, max_batches=20):
    model.to(device)
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            n_batches += 1
            if max_batches > 0 and n_batches >= max_batches:
                break
    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity


# ---------------------------------------------------------------------------
# Helpers: PyTorch <-> Flower Parameters
# ---------------------------------------------------------------------------


def get_parameters(model: nn.Module) -> Parameters:
    weights = [p.detach().cpu().numpy() for p in model.parameters()]
    return ndarrays_to_parameters(weights)


def set_parameters(model: nn.Module, parameters: Parameters):
    ndarrays = parameters_to_ndarrays(parameters)
    for param, ndarray in zip(model.parameters(), ndarrays):
        param.data = torch.from_numpy(ndarray).to(param.device)


# ---------------------------------------------------------------------------
# Main: serverless federated training
# ---------------------------------------------------------------------------


def main():
    # Config (matches Flower app defaults)
    num_nodes = 2
    num_rounds = 5
    max_steps = 50
    batch_size = 64
    block_size = 256
    lr = 5e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Nodes: {num_nodes}, Rounds: {num_rounds}, Max steps/round: {max_steps}")

    meta = _get_meta()

    gpt_config = GPTConfig(
        block_size=block_size,
        vocab_size=meta["vocab_size"],
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.2,
    )

    # Create per-node data loaders and models
    trainloaders = []
    valloaders = []
    models = []
    for i in range(num_nodes):
        tl, vl = load_data(i, num_nodes, batch_size, block_size)
        trainloaders.append(tl)
        valloaders.append(vl)
        models.append(GPT(gpt_config))

    # Shared storage and federated nodes
    shared_folder = InMemoryFolder()
    strategy = FedAvg()
    nodes = [
        AsyncFederatedNode(shared_folder=shared_folder, strategy=strategy)
        for _ in range(num_nodes)
    ]

    # Evaluate before training
    val_loss, ppl = test(models[0], valloaders[0], device)
    print(f"[Before training] val_loss={val_loss:.4f}  perplexity={ppl:.2f}")

    # Federated training loop
    for rnd in range(num_rounds):
        print(f"\n{'='*50}")
        print(f"Round {rnd + 1}/{num_rounds}")
        print(f"{'='*50}")

        for i in range(num_nodes):
            # Local training
            avg_loss = train(
                models[i], trainloaders[i], epochs=1, lr=lr,
                device=device, max_steps=max_steps,
            )
            print(f"  Node {i}: train_loss={avg_loss:.4f}")

            # Extract weights -> federation -> set weights back
            params = get_parameters(models[i])
            num_examples = min(max_steps, len(trainloaders[i])) * batch_size
            updated_params, _ = nodes[i].update_parameters(
                params, num_examples=num_examples,
            )
            if updated_params is not None:
                set_parameters(models[i], updated_params)
                print(f"  Node {i}: received aggregated parameters")
            else:
                print(f"  Node {i}: waiting for other nodes")

        # Evaluate after this round (use node 0's model)
        val_loss, ppl = test(models[0], valloaders[0], device)
        print(f"  [Round {rnd + 1}] val_loss={val_loss:.4f}  perplexity={ppl:.2f}")

    # Final evaluation
    print(f"\n{'='*50}")
    print("Final evaluation")
    print(f"{'='*50}")
    val_loss, ppl = test(models[0], valloaders[0], device)
    print(f"val_loss={val_loss:.4f}  perplexity={ppl:.2f}")

    # Generate a Shakespeare sample
    itos = meta["itos"]
    models[0].eval()
    models[0].to(device)
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = models[0].generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
    text = "".join(itos[i] for i in output[0].tolist())
    print("\n--- Generated Shakespeare sample ---")
    print(text)
    print("--- End of sample ---")


if __name__ == "__main__":
    main()
