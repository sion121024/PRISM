"""
학습된 이중언어 PRISM 체크포인트 로드 + 생성 (로컬 추론).

stage23_bilingual_fluent.py --save 로 저장한 체크포인트와
ByteLevel BPE 토크나이저(vocab.json + merges.txt)를 로드해 생성한다.

사용:
  python sample_bilingual.py --ckpt bilingual_fluent.pt --tok_dir . \
      --prompt "오늘 날씨는" --temp 0.7 --top_k 40 --n 100
"""

import argparse
import torch
from prism import PRISMLangModel


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    model = PRISMLangModel(
        vocab_size=a.get("vocab_real", a["vocab"]),
        d=a["d"], emb_dim=a["emb_dim"], K=a["K"],
        alpha=0.05, lam=0.01, memory_mode="sliding",
        mem_rank=a["mem_rank"], mem_scale=4.0,
        decoder="mlp", dec_hidden=a["d"], simple_prior=True, use_urec=True,
        use_gate=True, use_conv=True, d_conv=4,
        approximate_grad=True, n_layers=a["n_layers"],
    )
    # vocab 크기 불일치 시 state_dict에서 추론
    emb_w = ckpt["model"]["embed.weight"]
    if emb_w.shape[0] != model.vocab_size:
        model = PRISMLangModel(
            vocab_size=emb_w.shape[0], d=a["d"], emb_dim=a["emb_dim"], K=a["K"],
            alpha=0.05, lam=0.01, memory_mode="sliding", mem_rank=a["mem_rank"],
            mem_scale=4.0, decoder="mlp", dec_hidden=a["d"], simple_prior=True,
            use_urec=True, use_gate=True, use_conv=True, d_conv=4,
            approximate_grad=True, n_layers=a["n_layers"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_tokenizer(tok_dir):
    from tokenizers import ByteLevelBPETokenizer
    import os
    return ByteLevelBPETokenizer(
        os.path.join(tok_dir, "vocab.json"),
        os.path.join(tok_dir, "merges.txt"))


def generate(model, tok, prompt, device, n=100, temp=0.7, top_k=40):
    ids = tok.encode(prompt).ids or [0]
    p = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(p, max_new_tokens=n, temperature=temp, top_k=top_k)
    return tok.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="bilingual_fluent.pt")
    ap.add_argument("--tok_dir", default=".")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = load_model(args.ckpt, device)
    tok = load_tokenizer(args.tok_dir)
    print(f"params={model.num_params():,} vocab={model.vocab_size}")

    prompts = ([args.prompt] if args.prompt else
               ["오늘 날씨는", "나는 어제", "한국의 수도는 서울이고",
                "The president said", "In the morning, she", "Artificial intelligence is"])
    for pr in prompts:
        for t in ([args.temp] if args.prompt else [0.6, 0.8]):
            txt = generate(model, tok, pr, device, args.n, t, args.top_k).replace("\n", " ⏎ ")
            print(f"[t={t}] {txt}")


if __name__ == "__main__":
    main()
