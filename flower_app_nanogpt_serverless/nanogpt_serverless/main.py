"""Serverless federated NanoGPT training on Shakespeare.

Simulates federated nodes in a single process using AsyncFederatedNode.
Default: InMemoryFolder (single-process demo).
Use --shared-folder /path for LocalFolder (multi-process/multi-machine).

Usage:
    python -m nanogpt_serverless.main
    python -m nanogpt_serverless.main --shared-folder /tmp/nanogpt_shared
"""

import argparse
import math

import torch
from flwr.server.strategy import FedAvg

from .serverless import AsyncFederatedNode, InMemoryFolder, LocalFolder
from .task import (
    GPT,
    GPTConfig,
    _get_meta,
    get_parameters,
    load_data,
    set_parameters,
    test,
    train,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Serverless Federated NanoGPT")
    parser.add_argument(
        "--shared-folder", type=str, default=None,
        help="Path to shared folder for LocalFolder (multi-process). "
             "If not set, uses InMemoryFolder (single-process demo).",
    )
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Nodes: {args.num_nodes}, Rounds: {args.num_rounds}, Max steps/round: {args.max_steps}")

    meta = _get_meta()

    gpt_config = GPTConfig(
        block_size=args.block_size,
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
    for i in range(args.num_nodes):
        tl, vl = load_data(i, args.num_nodes, args.batch_size, args.block_size)
        trainloaders.append(tl)
        valloaders.append(vl)
        models.append(GPT(gpt_config))

    # Shared storage and federated nodes
    if args.shared_folder:
        print(f"Using LocalFolder at: {args.shared_folder}")
        shared_folder = LocalFolder(directory=args.shared_folder)
    else:
        print("Using InMemoryFolder (single-process demo)")
        shared_folder = InMemoryFolder()

    strategy = FedAvg()
    nodes = [
        AsyncFederatedNode(shared_folder=shared_folder, strategy=strategy)
        for _ in range(args.num_nodes)
    ]

    # Evaluate before training
    val_loss, ppl = test(models[0], valloaders[0], device)
    print(f"[Before training] val_loss={val_loss:.4f}  perplexity={ppl:.2f}")

    # Federated training loop
    for rnd in range(args.num_rounds):
        print(f"\n{'='*50}")
        print(f"Round {rnd + 1}/{args.num_rounds}")
        print(f"{'='*50}")

        for i in range(args.num_nodes):
            # Local training
            avg_loss = train(
                models[i], trainloaders[i], epochs=1, lr=args.lr,
                device=device, max_steps=args.max_steps,
            )
            print(f"  Node {i}: train_loss={avg_loss:.4f}")

            # Extract weights -> federation -> set weights back
            params = get_parameters(models[i])
            num_examples = min(args.max_steps, len(trainloaders[i])) * args.batch_size
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
