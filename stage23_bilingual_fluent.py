"""
Stage 4 (유창성): PRISM 한국어 + 영어 — ByteLevel BPE 서브워드.

설계철학(에너지 하강·이중시계·Hebbian 기억)은 그대로. 입력 단위만
바이트 → ByteLevel BPE 서브워드로 바꿔 유창성을 끌어올린다:
  - ByteLevel BPE = 바이트 기반(OOV 없음, 언어 무관 — 철학 유지)
    + 빈출 시퀀스를 서브워드로 병합 → 시퀀스 4~6배 짧아짐(문맥↑, 학습 효율↑).
  - 한국어는 글자당 3바이트라 바이트 단위가 특히 불리 → 서브워드 효과 큼.

데이터 (Kaggle 마운트): 한국어 kcbert + 영어 enwik8.
지표: 언어별 val perplexity(토큰). 생성: 한·영 프롬프트.

실행:
  python stage23_bilingual_fluent.py --smoke
  python stage23_bilingual_fluent.py --device cuda --d 640 --vocab 12000 --ko_mb 25 --en_mb 25
"""

import argparse
import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from prism import PRISMLangModel


# ------------------------------------------------------------------ #
# 데이터 로딩 (바이트 → 텍스트)                                       #
# ------------------------------------------------------------------ #

def find_largest(substrs, exts=(".txt",)):
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None
    best, best_sz = None, -1
    for dp, _, files in os.walk(root):
        for f in files:
            full = os.path.join(dp, f); low = full.lower()
            if any(s in low for s in substrs) and low.endswith(exts):
                try:
                    sz = os.path.getsize(full)
                except OSError:
                    continue
                if sz > best_sz:
                    best, best_sz = full, sz
    return best


def read_text(path, n_bytes, skip=1_000_000):
    with open(path, "rb") as f:
        f.seek(skip)
        return f.read(n_bytes).decode("utf-8", errors="ignore")


def collect_files(substr, n_bytes, name_prefix=None):
    """경로에 substr 포함된 파일들을 정렬·연결해 n_bytes까지 읽음 (다중 파일)."""
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None
    paths = []
    for dp, _, files in os.walk(root):
        for f in files:
            full = os.path.join(dp, f)
            if substr in full.lower() and (name_prefix is None or f.startswith(name_prefix)):
                if f.lower().endswith((".txt", ".tokens", ".raw")) or name_prefix:
                    paths.append(full)
    if not paths:
        return None
    paths.sort()
    buf, got = [], 0
    for p in paths:
        with open(p, "rb") as f:
            chunk = f.read(n_bytes - got)
        buf.append(chunk); got += len(chunk)
        if got >= n_bytes:
            break
    return b"".join(buf).decode("utf-8", errors="ignore")


def clean_en(t):
    # WikiText 토큰화 아티팩트 정리
    return (t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
             .replace(" @-@", "-"))


def load_korean(n_bytes):
    p = find_largest(["kcbert", "korean", "namu"])
    if p:
        print(f"Korean: {p}", flush=True)
        return read_text(p, n_bytes)
    return ("안녕하세요. 오늘 날씨가 좋네요. 한국어 모델을 학습합니다. "
            "나는 학생이고 책을 읽는 것을 좋아합니다. " * 8000)[: n_bytes // 2]


def load_english(n_bytes):
    # 우선순위: Simple English Wiki(쉬움·깨끗) > WikiText-103 > enwik8 > github
    t = collect_files("simpleenglish", n_bytes, name_prefix="wiki_")
    if t:
        print("English: Simple English Wikipedia", flush=True)
        return t
    p = find_largest(["wiki.train"], (".tokens", ".txt", ".raw"))
    if p:
        print(f"English: {p}", flush=True)
        return clean_en(read_text(p, n_bytes, skip=300_000))
    p = find_largest(["enwik8-train"]) or find_largest(["enwik8"])
    if p:
        print(f"English: {p}", flush=True)
        return read_text(p, n_bytes)
    try:
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
               "data/tinyshakespeare/input.txt")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return (r.read() * 4).decode("utf-8", errors="ignore")[: n_bytes]
    except Exception:
        return ("The quick brown fox jumps over the lazy dog. " * 8000)[: n_bytes]


# ------------------------------------------------------------------ #
# 토크나이저 (ByteLevel BPE)                                          #
# ------------------------------------------------------------------ #

def build_tokenizer(corpus_text, vocab_size, cache="/tmp/bbpe"):
    from tokenizers import ByteLevelBPETokenizer
    txt_path = "/tmp/bpe_corpus.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(corpus_text)
    tok = ByteLevelBPETokenizer()
    tok.train(files=[txt_path], vocab_size=vocab_size, min_frequency=2,
              special_tokens=["<pad>", "<bos>", "<eos>", "<ko>", "<en>"])
    return tok


# ------------------------------------------------------------------ #
# 데이터셋                                                            #
# ------------------------------------------------------------------ #

class TokenDataset(Dataset):
    def __init__(self, ids, block_size, stride=None, tag_id=None):
        stride = stride or block_size
        t = torch.tensor(ids, dtype=torch.long)
        if tag_id is None:
            self.chunks = [t[i:i + block_size + 1]
                           for i in range(0, len(t) - block_size - 1, stride)]
        else:
            # 각 청크 앞에 언어 태그를 붙여 생성 시 언어를 앵커링 (코드스위칭 억제)
            tg = torch.tensor([tag_id], dtype=torch.long)
            self.chunks = [torch.cat([tg, t[i:i + block_size]])
                           for i in range(0, len(t) - block_size, stride)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return self.chunks[i]


def evaluate(model, loader, device):
    model.eval(); tot, nb = 0.0, 0
    with torch.no_grad():
        for tokens in loader:
            tokens = tokens.to(device)
            tot += model(tokens)["loss"].item(); nb += 1
    return math.exp(tot / max(nb, 1))   # perplexity


def sample(model, tok, seed, device, n=60, temperature=0.6, top_k=40, rep=1.3,
           tag_id=None):
    model.eval()
    ids = tok.encode(seed).ids or [0]
    if tag_id is not None:
        ids = [tag_id] + ids                 # 언어 태그로 앵커링
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(prompt, max_new_tokens=n, temperature=temperature,
                         top_k=top_k, repetition_penalty=rep)
    gen = out[0].tolist()
    if tag_id is not None and gen and gen[0] == tag_id:
        gen = gen[1:]                        # 태그 토큰은 디코드에서 제외
    return tok.decode(gen)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--d", type=int, default=640)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--mem_rank", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--tbptt", type=int, default=64)
    p.add_argument("--vocab", type=int, default=12000)
    p.add_argument("--ko_mb", type=int, default=25)
    p.add_argument("--en_mb", type=int, default=25)
    p.add_argument("--lang_tags", type=int, default=1, help="언어 태그 토큰 앵커링(1/0)")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--save", default="")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.device = "cpu"; args.d = 96; args.emb_dim = 48; args.K = 2
        args.n_layers = 1; args.mem_rank = 8; args.block_size = 32
        args.batch_size = 16; args.epochs = 1; args.vocab = 500
        args.ko_mb = 1; args.en_mb = 1; args.tbptt = 16; args.max_steps = 5
        args.amp = False

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    device = torch.device(args.device)

    print("=" * 64)
    print("Stage 4 (유창성): PRISM 한국어+영어 — ByteLevel BPE 서브워드")
    print("=" * 64)
    ko = load_korean(args.ko_mb * 1_000_000)
    en = load_english(args.en_mb * 1_000_000)
    print(f"한국어 {len(ko):,}자 | 영어 {len(en):,}자 | device={device}", flush=True)

    # 공유 토크나이저 (한+영 결합 코퍼스로 학습)
    t0 = time.time()
    tok = build_tokenizer(ko + "\n" + en, args.vocab)
    vocab = tok.get_vocab_size()
    print(f"ByteLevel BPE vocab={vocab} ({time.time()-t0:.0f}s)", flush=True)

    ko_tag = tok.token_to_id("<ko>") if args.lang_tags else None
    en_tag = tok.token_to_id("<en>") if args.lang_tags else None

    def split_ids(text, frac=0.98):
        ids = tok.encode(text).ids
        n = int(len(ids) * frac)
        return ids[:n], ids[n:]
    ko_tr, ko_va = split_ids(ko)
    en_tr, en_va = split_ids(en)
    print(f"train_tokens={len(ko_tr)+len(en_tr):,} | ko_val={len(ko_va):,} | "
          f"en_val={len(en_va):,} | lang_tags={args.lang_tags}", flush=True)

    # 언어별 태그 청크를 만들어 합침 (각 청크가 자기 언어 태그로 시작)
    ko_ds = TokenDataset(ko_tr, args.block_size, tag_id=ko_tag)
    en_ds = TokenDataset(en_tr, args.block_size, tag_id=en_tag)
    train_ds = torch.utils.data.ConcatDataset([ko_ds, en_ds])
    ko_val = TokenDataset(ko_va, args.block_size, tag_id=ko_tag)
    en_val = TokenDataset(en_va, args.block_size, tag_id=en_tag)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    ko_loader = DataLoader(ko_val, batch_size=args.batch_size)
    en_loader = DataLoader(en_val, batch_size=args.batch_size)
    print(f"train_chunks={len(train_ds):,}", flush=True)

    model = PRISMLangModel(
        vocab_size=vocab, d=args.d, emb_dim=args.emb_dim, K=args.K,
        alpha=0.05, lam=0.01, memory_mode="sliding",
        mem_rank=args.mem_rank, mem_scale=4.0,
        decoder="mlp", dec_hidden=args.d, simple_prior=True, use_urec=True,
        use_gate=True, use_conv=True, d_conv=4,
        approximate_grad=True, n_layers=args.n_layers,
    ).to(device)
    print(f"PRISM params: {model.num_params():,}", flush=True)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
    use_amp = (device.type == "cuda" and args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train(); ep_loss, nb, t0 = 0.0, 0, time.time()
        for tokens in train_loader:
            tokens = tokens.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(tokens, tbptt_window=args.tbptt)["loss"]
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ep_loss += loss.item(); nb += 1
            if args.max_steps and nb >= args.max_steps:
                break
        ko_ppl = evaluate(model, ko_loader, device)
        en_ppl = evaluate(model, en_loader, device)
        mem = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
        print(f"ep {epoch:2d} | loss {ep_loss/nb:.4f} | 한국어 ppl {ko_ppl:.2f} | "
              f"영어 ppl {en_ppl:.2f} | {time.time()-t0:.0f}s | {mem:.1f}GB", flush=True)
        print("  [KO] " + sample(model, tok, "오늘", device, 50, tag_id=ko_tag).replace("\n", " ⏎ "), flush=True)
        print("  [EN] " + sample(model, tok, "The ", device, 50, tag_id=en_tag).replace("\n", " ⏎ "), flush=True)

    print("\n" + "=" * 64 + "\n최종 생성 샘플:")
    for s in ["오늘 날씨는", "나는 어제", "한국의 수도는"]:
        print(f"  [KO '{s}'] " + sample(model, tok, s, device, 80, tag_id=ko_tag).replace("\n", " ⏎ "))
    for s in ["The president", "In the morning", "She said that"]:
        print(f"  [EN '{s}'] " + sample(model, tok, s, device, 80, tag_id=en_tag).replace("\n", " ⏎ "))

    if args.save:
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.save)
        tok.save_model("/kaggle/working")
        print(f"saved → {args.save}")


if __name__ == "__main__":
    main()
