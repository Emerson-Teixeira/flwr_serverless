# FlowerHub App Plan: NanoGPT Shakespeare (Federated)

## Goal

Submit a FlowerHub app that demonstrates federated learning with NanoGPT on the
Tiny Shakespeare dataset (character-level tokenizer, CPU-only). Standard Flower
(not serverless) for now — modernize `flwr_serverless` later.

---

## App Structure

```
nanogpt-shakespeare/
├── pyproject.toml
├── README.md
├── nanogpt_shakespeare/
│   ├── __init__.py
│   ├── client_app.py      # ClientApp: train / evaluate
│   ├── server_app.py      # ServerApp: FedAvg orchestration
│   └── task.py            # NanoGPT model, data loading, train/test fns
```

---

## Step-by-Step Plan

### 1. Create the app directory and pyproject.toml

Create `nanogpt-shakespeare/` at the repo root with a `pyproject.toml` following
the FlowerHub convention:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "nanogpt-shakespeare"
version = "1.0.0"
description = "Federated NanoGPT on Tiny Shakespeare with Flower"
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.26.1",
    "torch>=2.0.0",
    "numpy",
    "requests",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "kungfuai"

[tool.flwr.app.components]
serverapp = "nanogpt_shakespeare.server_app:app"
clientapp = "nanogpt_shakespeare.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 5
local-epochs = 1
learning-rate = 1e-3
batch-size = 64
block-size = 256
# Baby GPT config (CPU-friendly)
n-layer = 6
n-head = 6
n-embd = 384
dropout = 0.2
```

### 2. Write `task.py` — model, data, train/test

Port the NanoGPT model from CVlization. Simplify heavily:

- **Model**: `GPT` class with `GPTConfig` dataclass. Keep `CausalSelfAttention`,
  `MLP`, `Block`, `LayerNorm`. Strip out GPT-2 weight loading, Flash Attention
  guards, and `torch.compile` — keep it minimal and CPU-friendly.
- **Data loading**: Download Tiny Shakespeare
  (`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`),
  build char-level vocab (stoi/itos), encode to int tensors. Cache locally.
- **Partitioning**: Split the text into N equal chunks (one per federated client).
  Each client gets a contiguous portion. Within each partition, use 90/10
  train/val split. Wrap in a simple `Dataset` that yields `(x, y)` pairs of
  `block_size` length.
- **`train(model, dataloader, epochs, lr, device)`**: Standard cross-entropy
  training loop with AdamW. Returns average train loss.
- **`test(model, dataloader, device)`**: Evaluate cross-entropy loss on val set.
  Returns loss and perplexity.

Key design decisions:
- No `flwr-datasets` / `FederatedDataset` dependency — Shakespeare is tiny and
  text-based, not in HuggingFace datasets format. We download and partition it
  ourselves.
- CPU-only: force `device = "cpu"`. No CUDA/MPS logic needed.
- Vocab (~65 chars) and meta info built at data-load time, shared across clients
  (deterministic from the same text).

### 3. Write `client_app.py`

Follow the quickstart-pytorch pattern exactly:

```python
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # 1. Read run_config for hyperparams
    # 2. Build model from config (n_layer, n_head, n_embd, etc.)
    # 3. Load state dict from msg.content["arrays"]
    # 4. Load this client's data partition using context.node_config["partition-id"]
    # 5. Train for local-epochs
    # 6. Return updated state dict + metrics (train_loss)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    # 1. Build model, load weights from msg
    # 2. Load this client's val partition
    # 3. Compute val loss and perplexity
    # 4. Return metrics
```

### 4. Write `server_app.py`

```python
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context):
    # 1. Read config (num rounds, lr, model config)
    # 2. Build initial GPT model (random init)
    # 3. Create FedAvg strategy
    # 4. Run federated rounds
    # 5. Optionally generate a sample from the final model
```

Nice touch: after training, generate a short Shakespeare-style text sample and
print it, so the user sees a tangible result.

### 5. Add `__init__.py` and `README.md`

- `__init__.py`: empty
- `README.md`: brief description, install/run instructions, what to expect

### 6. Test locally

```bash
cd nanogpt-shakespeare
pip install -e .
flwr run .
```

Verify:
- Data downloads and partitions correctly
- Multiple simulated clients train and report loss
- Server aggregates and loss decreases over rounds
- Final model generates semi-coherent Shakespeare text

### 7. Publish to FlowerHub

```bash
flwr login          # authenticate
flwr app publish .  # upload
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| **Flower API mismatch** — the `flwr.app`, `flwr.clientapp`, `flwr.serverapp` imports may differ between Flower versions | Pin to the exact version in `../flower` repo (`>=1.26.1`). Install from local `../flower` if needed to test. |
| **Model size** — even baby GPT has ~10M params; FedAvg on CPU with multiple clients may be slow | Keep defaults small: 6 layers, 384 dim, 5 rounds. Simulation runs sequentially anyway. |
| **Data partitioning** — contiguous text splits mean each client sees different "styles" (early vs late Shakespeare) | Acceptable for a demo. Could shuffle sentences later. |
| **No `flwr-datasets`** — FlowerHub apps typically use it | Not required. The quickstart examples use it for convenience, but raw data loading is fine. |

---

## Estimated Effort

- **task.py** (model + data): ~200 lines — largest piece, but mostly ported from CVlization
- **client_app.py**: ~60 lines
- **server_app.py**: ~50 lines
- **pyproject.toml + README**: ~50 lines
- **Testing & debugging**: depends on API quirks

Total: a focused session of work.

---

## Future: Serverless Version

Once this standard Flower app works and is published, the next step is:
1. Port `flwr_serverless` to `flwr>=1.26`
2. Create a second FlowerHub app (`nanogpt-shakespeare-serverless`) that replaces
   FedAvg orchestration with `AsyncFederatedNode` + shared storage
3. This demonstrates the value prop: same model, same data, but no central server
