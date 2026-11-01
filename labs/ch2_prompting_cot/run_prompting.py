# -*- coding: utf-8 -*-
"""ch2 · 提示学习与思维链：在 GSM8K 测试集子集上比较四种提示策略（DeepSeek API）。

课程原 notebook 走 OpenAI 兼容接口；本机可用的是 DeepSeek（`https://api.deepseek.com`、`deepseek-chat`），
故改为同一套 OpenAI 兼容协议 + 环境变量里的 key。

四种策略：zero-shot 直答 / zero-shot CoT / few-shot 直答 / few-shot CoT。
指标：最终数值答案的 exact-match、平均输出 token、总调用次数。

运行前设置：$env:DEEPSEEK_API_KEY="sk-..."
产出：results/metrics.json、results/per_question.csv、results/log.txt
"""
import concurrent.futures as cf
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "https://api.deepseek.com"
MODEL = "deepseek-chat"
KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
N_QUESTIONS = 40
GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
             "grade_school_math/data/test.jsonl")

FEW_SHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
     "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. After planting there are 21 trees, so the workers planted 21 - 15 = 6 trees. "
     "The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are 3 cars. 2 more arrive, so 3 + 2 = 5 cars. The answer is 5."),
]


def log(msg, path):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_gsm8k():
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "gsm8k_test.jsonl")
    if not os.path.exists(path):
        raw = urllib.request.urlopen(GSM8K_URL, timeout=60).read().decode("utf-8")
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            gold = d["answer"].split("####")[-1].strip().replace(",", "")
            items.append({"question": d["question"].strip(), "gold": gold})
    rng = random.Random(2026)
    rng.shuffle(items)
    return items[:N_QUESTIONS]


def build_messages(strategy, question):
    if strategy == "zero_shot":
        return [{"role": "user", "content": question + "\nAnswer with a single number."}]
    if strategy == "cot":
        return [{"role": "user", "content": question + "\nThink step by step, then end with 'The answer is <number>'."}]
    shots = "\n\n".join("Q: %s\nA: %s" % (q, a) for q, a in FEW_SHOT)
    if strategy == "few_shot":
        return [{"role": "user", "content": shots + "\n\nQ: " + question + "\nA: Answer with a single number."}]
    return [{"role": "user", "content": shots + "\n\nQ: " + question +
             "\nA: Think step by step, then end with 'The answer is <number>'."}]


def call(messages, max_tokens=512, retries=3):
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0,
                       "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except urllib.error.HTTPError as e:
            last = "HTTP %s %s" % (e.code, e.read()[:120].decode("utf-8", "replace"))
        except Exception as e:
            last = "%s %s" % (type(e).__name__, str(e)[:100])
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(last)


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text):
    hits = NUM_RE.findall(text.replace("$", ""))
    if not hits:
        return None
    s = hits[-1].replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else "%.2f" % v


def norm(x):
    try:
        v = float(str(x).replace(",", ""))
        return str(int(v)) if v == int(v) else "%.2f" % v
    except Exception:
        return str(x)


def main():
    if not KEY:
        print("缺少 DEEPSEEK_API_KEY 环境变量", file=sys.stderr)
        return 2
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    log_path = os.path.join(HERE, "results", "log.txt")
    items = load_gsm8k()
    log("questions=%d model=%s" % (len(items), MODEL), log_path)

    strategies = ["zero_shot", "cot", "few_shot", "few_shot_cot"]
    rows, metrics = [], {}
    for st in strategies:
        t0 = time.time()
        ok, tokens_out, tokens_in = 0, 0, 0

        def work(i):
            q = items[i]["question"]
            text, usage = call(build_messages(st, q))
            return i, text, usage

        results = {}
        with cf.ThreadPoolExecutor(max_workers=6) as pool:
            for i, text, usage in pool.map(work, range(len(items))):
                results[i] = (text, usage)

        for i in range(len(items)):
            text, usage = results[i]
            pred = extract_number(text)
            gold = norm(items[i]["gold"])
            correct = pred is not None and pred == gold
            ok += int(correct)
            tokens_out += usage.get("completion_tokens", 0)
            tokens_in += usage.get("prompt_tokens", 0)
            rows.append({"strategy": st, "idx": i, "gold": gold, "pred": pred or "",
                         "correct": int(correct), "completion_tokens": usage.get("completion_tokens", 0)})
        acc = ok / len(items)
        metrics[st] = {"accuracy": round(acc, 4), "correct": ok, "n": len(items),
                       "mean_completion_tokens": round(tokens_out / len(items), 1),
                       "prompt_tokens_total": tokens_in,
                       "seconds": round(time.time() - t0, 1)}
        log("%s: acc=%.3f (%d/%d) mean_out_tokens=%.1f" % (st, acc, ok, len(items), tokens_out / len(items)), log_path)

    with open(os.path.join(HERE, "results", "per_question.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, "results", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "n_questions": len(items), "strategies": metrics}, f,
                  ensure_ascii=False, indent=2)
    log("done: %s" % json.dumps(metrics, ensure_ascii=False), log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
