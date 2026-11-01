# -*- coding: utf-8 -*-
"""按 seed 逐个跑 ch1，汇总 mean±sd 到 results/seeds_summary.json。"""
import json
import os
import subprocess
import sys

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 43, 44]

rows = []
for s in SEEDS:
    out = os.path.join("results", "seed%d" % s)
    os.makedirs(out, exist_ok=True)
    cmd = [PY, "train_bert.py", "--epochs", "1", "--batch", "8", "--accum", "4", "--max_len", "512",
           "--seed", str(s), "--out", out]
    print("RUN", s, flush=True)
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("seed %d FAILED\n%s" % (s, (r.stderr or "")[-2000:]), flush=True)
        continue
    with open(os.path.join(HERE, out, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    rows.append(m)
    print("seed %d val_acc=%.4f" % (s, m["val_accuracy"]), flush=True)

if rows:
    accs = [m["val_accuracy"] for m in rows]
    mean = sum(accs) / len(accs)
    var = sum((x - mean) ** 2 for x in accs) / max(len(accs) - 1, 1)
    summary = {
        "seeds": [m["seed"] for m in rows],
        "val_accuracy": accs,
        "mean": round(mean, 4),
        "sd": round(var ** 0.5, 4),
        "seconds_each": [m["seconds"] for m in rows],
    }
    with open(os.path.join(HERE, "results", "seeds_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
