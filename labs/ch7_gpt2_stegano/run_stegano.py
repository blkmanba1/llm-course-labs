# -*- coding: utf-8 -*-
"""ch7 · GPT-2 文本隐写：编码 + 解码验证。

来源：documents/chapter7/llm_stega.ipynb（课程原实现，只做了编码，没有给出解码正确率）。
本脚本在其基础上补三件事：
  1) 把生成上限从 30 步提到 200 步（notebook 默认只嵌入约 56 比特）；
  2) 实现对应的解码器（同 prompt、同模型、逐位重算），给出比特恢复率；
  3) 保留隐写文本与普通文本，便于人工比对。

产出：results/outputs-gpt2-{stega,normal}-{huffman,flc}.txt、results/metrics.json、results/log.txt
"""
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MAX_STEPS = 200


class Node:
    def __init__(self, freq):
        self.left = self.right = self.father = None
        self.freq = freq

    def isLeft(self):
        return self.father.left is self


def _nodes(freqs):
    return [Node(f) for f in freqs]


def _tree(nodes):
    q = nodes[:]
    while len(q) > 1:
        q.sort(key=lambda n: n.freq)
        l, r = q.pop(0), q.pop(0)
        p = Node(l.freq + r.freq)
        p.left, p.right = l, r
        l.father = r.father = p
        q.append(p)
    q[0].father = None
    return q[0]


def _codes(nodes, root):
    out = [""] * len(nodes)
    for i, n in enumerate(nodes):
        cur = n
        while cur is not root:
            out[i] = ("0" if cur.isLeft() else "1") + out[i]
            cur = cur.father
    return out


def huffman_table(pred_prob, k):
    prob, idx = torch.topk(pred_prob, k=k, dim=1)
    pairs = [[i.item(), j.item()] for i, j in zip(idx[0], prob[0])]
    nodes = _nodes([p[1] for p in pairs])
    codes = _codes(nodes, _tree(nodes))
    return [(pairs[i][0], codes[i]) for i in range(len(pairs))]


def flc_table(pred_prob, k):
    prob, idx = torch.topk(pred_prob, k=2 ** k, dim=1)
    ids = idx[0].tolist()
    codes = [bin(i)[2:].zfill(k) for i in range(2 ** k)]
    return list(zip(ids, codes))


def table_for(probs, scheme, k=2):
    return huffman_table(probs, 2 ** k) if scheme == "huffman" else flc_table(probs, k)


@torch.no_grad()
def encode(model, tok, prompt, bits, scheme, k=2, max_steps=MAX_STEPS):
    device = model.device
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    remaining = bits
    used = 0
    for _ in range(max_steps):
        if not remaining:
            break
        probs = F.softmax(model(ids).logits[:, -1, :], dim=-1)
        chosen = None
        for tid, code in table_for(probs, scheme, k):
            if remaining.startswith(code):
                chosen = (tid, code)
                break
        if chosen is None:
            break
        ids = torch.cat([ids, torch.LongTensor([[chosen[0]]]).to(device)], dim=1)
        remaining = remaining[len(chosen[1]):]
        used += len(chosen[1])
    return ids, used


@torch.no_grad()
def decode(model, tok, prompt, ids, scheme, k=2):
    p_len = len(tok.encode(prompt))
    bits = []
    for t in range(p_len, ids.shape[1]):
        probs = F.softmax(model(ids[:, :t]).logits[:, -1, :], dim=-1)
        want = int(ids[0, t].item())
        hit = [c for tid, c in table_for(probs, scheme, k) if tid == want]
        if not hit:
            break
        bits.append(hit[0])
    return "".join(bits)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def bit_accuracy(sent, recovered):
    n = min(len(sent), len(recovered))
    if n == 0:
        return 0.0, 0
    return sum(1 for i in range(n) if sent[i] == recovered[i]) / n, n


def main():
    os.makedirs("results", exist_ok=True)
    log_path = "results/log.txt"

    def log(m):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    setup_seed(2023)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log("device=%s max_steps=%d" % (device, MAX_STEPS))
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    log("gpt2 params=%.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    out = {}
    for scheme in ("huffman", "flc"):
        bits = "".join(random.choice("01") for _ in range(3000))
        t0 = time.time()
        ids, used = encode(model, tok, "Once upon a time", bits, scheme)
        enc_s = time.time() - t0
        recovered = decode(model, tok, "Once upon a time", ids, scheme)
        acc, cmp_len = bit_accuracy(bits[:used], recovered)
        n_tok = int(ids.shape[1] - len(tok.encode("Once upon a time")))
        text = tok.decode(ids[0], skip_special_tokens=True)
        out[scheme] = {
            "bits_sent": used,
            "tokens_generated": n_tok,
            "bits_per_token": round(used / max(n_tok, 1), 2),
            "decode_bit_accuracy": round(acc, 4),
            "decode_bits_compared": cmp_len,
            "encode_seconds": round(enc_s, 1),
        }
        with open("results/outputs-gpt2-stega-%s.txt" % scheme, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        log("%s: %s" % (scheme, json.dumps(out[scheme], ensure_ascii=False)))
        log("  text: %s" % text[:140].replace("\n", " "))

    setup_seed(2023)
    ids = tok.encode("Once upon a time", return_tensors="pt").to(device)
    for _ in range(MAX_STEPS):
        probs = F.softmax(model(ids).logits[:, -1, :], dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt[0].item() == tok.eos_token_id:
            break
    normal = tok.decode(ids[0], skip_special_tokens=True)
    with open("results/outputs-gpt2-normal.txt", "w", encoding="utf-8") as f:
        f.write(normal + "\n")
    log("normal: %s" % normal[:140].replace("\n", " "))

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("done")


if __name__ == "__main__":
    main()
