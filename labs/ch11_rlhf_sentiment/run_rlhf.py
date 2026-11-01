# -*- coding: utf-8 -*-
"""ch11 · 用情感奖励做 RLHF：REINFORCE + KL 约束（课程第 11 章的极简替代实现）。

课程原版用 trl 的 PPOTrainer（`gpt2-imdb` 作策略、`distilbert-imdb` 作奖励），
但那套 API 与本机 transformers 5.8 不兼容，故用同样的两个模型重写为一个最小策略梯度循环：

  loss = -(reward - baseline) * sum(logp_policy) + kl_coef * sum(logp_policy - logp_ref)

产出：results/metrics.json（更新前后平均奖励、KL、样例文本）、results/log.txt
"""
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROMPTS = [
    "This movie", "I watched this film", "The story", "The acting",
    "The ending", "This film", "The dialogue", "I think this movie",
]
STEPS = 60
BATCH = 4
GEN_LEN = 20
KL_COEF = 0.02
LR = 1e-5


def log(msg, path):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@torch.no_grad()
def reward_of(texts, rm_tok, rm, device):
    enc = rm_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    logits = rm(**enc).logits
    probs = F.softmax(logits, dim=-1)
    pos = rm.config.label2id.get("POSITIVE", 1)
    return probs[:, pos].tolist()


def token_logprobs(model, input_ids, gen_start):
    out = model(input_ids=input_ids)
    logp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
    tgt = input_ids[:, 1:]
    gathered = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    gen = gathered[:, gen_start - 1:]
    return gen.sum(-1), gen


def main():
    os.makedirs("results", exist_ok=True)
    log_path = "results/log.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device=%s steps=%d batch=%d gen_len=%d kl=%.3f" % (device, STEPS, BATCH, GEN_LEN, KL_COEF), log_path)

    tok = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
    tok.pad_token = tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb").to(device)
    ref = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb").to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    rm_tok = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    rm = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb").to(device).eval()
    log("models loaded, reward labels=%s" % rm.config.id2label, log_path)

    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    baseline = None
    rewards_hist = []
    kl_hist = []

    policy.train()
    for step in range(STEPS):
        batch = [random.choice(PROMPTS) for _ in range(BATCH)]
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        gen_start = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = policy.generate(**enc, max_new_tokens=GEN_LEN, do_sample=True, top_k=50, top_p=0.95,
                                  pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(out, skip_special_tokens=True)
        rewards = torch.tensor(reward_of(texts, rm_tok, rm, device), device=device)

        adv, _ = token_logprobs(policy, out, gen_start)
        with torch.no_grad():
            adv_ref, _ = token_logprobs(ref, out, gen_start)
        b = rewards.mean() if baseline is None else baseline
        loss = -((rewards - b) * adv).mean() + KL_COEF * (adv - adv_ref).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        r = rewards.mean().item()
        baseline = r if baseline is None else 0.9 * baseline + 0.1 * r
        rewards_hist.append(r)
        kl_hist.append((adv - adv_ref).mean().item())

        if step % 10 == 0 or step == STEPS - 1:
            log("step %d reward %.4f baseline %.4f kl %.4f s0=%s" %
                (step, r, baseline, kl_hist[-1], texts[0][:80].replace("\n", " ")), log_path)

    # 训练后评估：固定 8 个提示，各采样 4 条，算平均奖励（与训练时同分布）
    policy.eval()
    evals, samples = [], []
    with torch.no_grad():
        for p in PROMPTS:
            enc = tok([p] * 4, return_tensors="pt", padding=True).to(device)
            out = policy.generate(**enc, max_new_tokens=GEN_LEN, do_sample=True, top_k=50, top_p=0.95,
                                  pad_token_id=tok.eos_token_id)
            texts = tok.batch_decode(out, skip_special_tokens=True)
            evals += reward_of(texts, rm_tok, rm, device)
            samples.append(texts[0])
    after = sum(evals) / len(evals)
    before = sum(rewards_hist[:5]) / 5
    metrics = {
        "policy": "lvwerra/gpt2-imdb", "reward_model": "lvwerra/distilbert-imdb",
        "steps": STEPS, "batch": BATCH, "gen_len": GEN_LEN, "kl_coef": KL_COEF, "lr": LR,
        "mean_reward_first5_steps": round(before, 4),
        "mean_reward_last5_steps": round(sum(rewards_hist[-5:]) / 5, 4),
        "mean_reward_eval_after": round(after, 4),
        "mean_kl": round(sum(kl_hist) / len(kl_hist), 4),
        "samples": samples[:4],
    }
    with open(os.path.join("results", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join("results", "samples.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(samples))
    log("metrics: %s" % json.dumps({k: v for k, v in metrics.items() if k != "samples"}, ensure_ascii=False), log_path)


if __name__ == "__main__":
    main()
