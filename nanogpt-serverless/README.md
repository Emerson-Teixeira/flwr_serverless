# Serverless Federated NanoGPT on Shakespeare

A character-level GPT trained on Tiny Shakespeare using **serverless federated learning** — no central Flower server required.

This app vendors `flwr_serverless` modules so it can be distributed as a standalone Flower Hub app without `pip install flwr_serverless`.

## Quick Start

```bash
cd nanogpt-serverless
pip install -e .
python -m nanogpt_serverless.main
```

This runs a single-process simulation with 2 federated nodes using `InMemoryFolder`.

## Multi-Process / Multi-Machine

Use `--shared-folder` to point at a directory accessible by all participants (e.g., NFS mount):

```bash
# Terminal 1
python -m nanogpt_serverless.main --shared-folder /tmp/nanogpt_shared

# Terminal 2 (on same or different machine)
python -m nanogpt_serverless.main --shared-folder /tmp/nanogpt_shared
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--shared-folder` | `None` | Path for `LocalFolder`. If unset, uses `InMemoryFolder`. |
| `--num-nodes` | `2` | Number of simulated federated nodes |
| `--num-rounds` | `5` | Number of federated rounds |
| `--max-steps` | `50` | Max training steps per round per node |
| `--batch-size` | `64` | Batch size |
| `--block-size` | `256` | Context window size |
| `--lr` | `5e-4` | Learning rate |

## Expected Results

Over 5 rounds, perplexity drops from ~67 to ~12, and the model generates Shakespeare-like text.
