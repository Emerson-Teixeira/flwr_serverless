# Federated NanoGPT on Tiny Shakespeare

Train a character-level GPT (NanoGPT) on the Tiny Shakespeare dataset using
federated learning with [Flower](https://flower.ai).

Each client receives a contiguous partition of the training text and trains a
small transformer locally. The server aggregates weights with FedAvg each round.
After training, the server generates a Shakespeare-style text sample.

## Results

With default settings (5 rounds, 2 clients, 50 steps/round, CPU):

| Round | Train Loss | Val Loss | Perplexity |
|-------|-----------|----------|------------|
| 0     | —         | 4.23     | 69.0       |
| 1     | 3.26      | 2.97     | 19.5       |
| 2     | 2.74      | 2.64     | 14.0       |
| 3     | 2.60      | 2.56     | 13.0       |
| 4     | 2.53      | 2.52     | 12.5       |
| 5     | 2.50      | 2.52     | 12.4       |

## Quickstart

```bash
pip install -e .
flwr run .
```

**Note**: `flwr run` requires Python <= 3.12 (Ray does not support 3.13 yet on
macOS). If you have Python 3.13, use the simulation script directly:

```bash
python run_sim.py
```

## Configuration

Override defaults via `--run-config`:

```bash
flwr run . --run-config "num-server-rounds=10 learning-rate=5e-4 local-epochs=2"
```

| Parameter           | Default | Description                          |
|---------------------|---------|--------------------------------------|
| num-server-rounds   | 5       | Number of federated rounds           |
| local-epochs        | 1       | Training epochs per client per round |
| learning-rate       | 1e-3    | AdamW learning rate                  |
| batch-size          | 64      | Training batch size                  |
| block-size          | 256     | Context window length (chars)        |
| max-steps           | 50      | Max training steps per client per round (0 = unlimited) |
| n-layer             | 6       | Transformer layers                   |
| n-head              | 6       | Attention heads                      |
| n-embd              | 384     | Embedding dimension                  |
| dropout             | 0.2     | Dropout rate                         |

## Model

A "baby GPT" (~10.8M parameters) based on
[Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), with weight tying,
causal self-attention, and GELU activations. Small enough to train on CPU.

## Dataset

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
(~1MB, 65-character vocabulary). Downloaded automatically on first run.
The training set is split evenly across clients as contiguous text chunks.

## How it works

```
┌─────────────┐     ┌─────────────┐
│  Client 0   │     │  Client 1   │
│ (Acts 1-3)  │     │ (Acts 4-5)  │
│  Local GPT  │     │  Local GPT  │
└──────┬──────┘     └──────┬──────┘
       │    model weights    │
       └────────┬────────────┘
          ┌─────┴─────┐
          │  Server   │
          │  FedAvg   │
          │ aggregate │
          └───────────┘
```

Each round:
1. Server sends global model weights to all clients
2. Each client trains on its local Shakespeare partition for `max-steps` steps
3. Clients send updated weights back to the server
4. Server averages the weights (FedAvg) and evaluates on the full validation set
5. After the final round, the server generates a text sample from the trained model

## Project structure

```
nanogpt-shakespeare/
├── pyproject.toml                    # Flower app config & dependencies
├── run_sim.py                        # Direct simulation script (no flwr CLI)
├── Dockerfile                        # Docker build for environments without Python 3.11
└── nanogpt_shakespeare/
    ├── __init__.py
    ├── task.py                       # GPT model, data loading, train/test
    ├── client_app.py                 # Flower ClientApp (train + evaluate)
    └── server_app.py                 # Flower ServerApp (FedAvg + text generation)
```
