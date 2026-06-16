"""
Generate text from a trained PRISM checkpoint.

Usage:
    python generate.py --ckpt checkpoints/prism.pt --prompt "HAMLET:"
    python generate.py --ckpt checkpoints/prism.pt --K 32   # deeper thinking
"""

import argparse
import torch
from prism import PRISMLM, CharDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--prompt", default="HAMLET:")
    parser.add_argument("--n",      type=int,   default=300)
    parser.add_argument("--temp",   type=float, default=0.8)
    parser.add_argument("--K",      type=int,   default=None,
                        help="override inner steps (more = deeper thinking)")
    parser.add_argument("--top_k",  type=int,   default=40)
    args = parser.parse_args()

    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg   = ckpt["args"]
    vocab = ckpt["vocab"]

    model = PRISMLM(
        vocab_size=len(vocab["stoi"]),
        d=cfg["d"],
        K=cfg["K"],
        step=cfg["step"],
    )
    model.load_state_dict(ckpt["model"])

    stoi = vocab["stoi"]
    itos = vocab["itos"]

    prompt_ids = torch.tensor(
        [stoi[c] for c in args.prompt if c in stoi], dtype=torch.long
    ).unsqueeze(0)

    generated = model.generate(
        prompt_ids,
        max_new=args.n,
        temperature=args.temp,
        K=args.K,
        top_k=args.top_k,
    )

    text = args.prompt + "".join(itos[i] for i in generated)
    print(text)


if __name__ == "__main__":
    main()
