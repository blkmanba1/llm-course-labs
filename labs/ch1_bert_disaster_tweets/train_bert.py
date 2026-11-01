# -*- coding: utf-8 -*-
"""ch1 · BERT 灾害推文分类：复现课程第一章的监督微调闭环。

数据：documents/chapter1/TextClassification/TextClassification/data/{train,val}.csv（test.csv 无标签）
设置：bert-base-uncased，1 epoch，lr 2e-5，有效 batch 32（8×4 梯度累积），max_len 512。
产出：results/metrics.json、results/log.txt、results/test_predictions.csv
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATA = os.environ.get(
    "CH1_DATA",
    r"data",
)


def read_split(name, labelled=True):
    df = pd.read_csv(os.path.join(DATA, name + ".csv"))
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    if labelled:
        df = df.dropna(subset=["target"]).reset_index(drop=True)
        df["target"] = df["target"].astype("int64")
    return df


class TweetSet(Dataset):
    def __init__(self, df, tok, max_len, labelled=True):
        self.text = df["text"].astype(str).tolist()
        self.y = df["target"].tolist() if labelled else None
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        enc = self.tok(self.text[i], truncation=True, max_length=self.max_len, padding="max_length")
        item = {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
        }
        if self.y is not None:
            item["labels"] = torch.tensor(self.y[i])
        return item


def log(msg, path):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@torch.no_grad()
def evaluate(model, loader, device, labelled=True):
    model.eval()
    preds, gold, loss_sum, n = [], [], 0.0, 0
    for b in loader:
        kw = {"input_ids": b["input_ids"].to(device), "attention_mask": b["attention_mask"].to(device)}
        if labelled:
            kw["labels"] = b["labels"].to(device)
        out = model(**kw)
        if labelled:
            loss_sum += out.loss.item() * len(b["labels"])
            n += len(b["labels"])
            gold += b["labels"].tolist()
        preds += out.logits.argmax(-1).cpu().tolist()
    acc = float(np.mean([p == g for p, g in zip(preds, gold)])) if gold else None
    return acc, (loss_sum / max(n, 1) if labelled else None), preds, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, "log.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device=%s %s" % (device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"), log_path)
    import random as _random
    _random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    log("seed=%d" % a.seed, log_path)

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

    tr = TweetSet(read_split("train"), tok, a.max_len)
    va = TweetSet(read_split("val"), tok, a.max_len)
    te = TweetSet(read_split("test", labelled=False), tok, a.max_len, labelled=False)
    log("rows train=%d val=%d test=%d(unlabelled) max_len=%d batch=%d accum=%d" %
        (len(tr), len(va), len(te), a.max_len, a.batch, a.accum), log_path)

    dl_tr = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=0)
    dl_va = DataLoader(va, batch_size=64, num_workers=0)
    dl_te = DataLoader(te, batch_size=64, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    steps = max(int(len(dl_tr) / a.accum * a.epochs), 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    model.train()
    t0 = time.time()
    step = 0
    for ep in range(int(a.epochs)):
        for i, b in enumerate(dl_tr):
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            (out.loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 100 == 0:
                    log("epoch %d step %d/%d loss %.4f" % (ep + 1, step, steps, out.loss.item()), log_path)
        acc, vloss, _, _ = evaluate(model, dl_va, device)
        log("epoch %d done val_acc=%.4f val_loss=%.4f elapsed=%.0fs" % (ep + 1, acc, vloss, time.time() - t0), log_path)
        model.train()

    val_acc, val_loss, _, _ = evaluate(model, dl_va, device)
    _, _, te_preds, _ = evaluate(model, dl_te, device, labelled=False)
    log("FINAL val_acc=%.4f val_loss=%.4f" % (val_acc, val_loss), log_path)

    _, _, preds, gold = evaluate(model, dl_va, device)
    tp = sum(1 for p, g in zip(preds, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, gold) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, gold) if p == 0 and g == 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    metrics = {
        "model": "bert-base-uncased",
        "seed": a.seed,
        "epochs": a.epochs, "lr": a.lr, "batch": a.batch, "accum": a.accum, "max_len": a.max_len,
        "rows": {"train": len(tr), "val": len(va), "test_unlabelled": len(te)},
        "val_accuracy": round(val_acc, 4),
        "val_precision_pos": round(prec, 4), "val_recall_pos": round(rec, 4), "val_f1_pos": round(f1, 4),
        "test_positive_rate": round(sum(te_preds) / max(len(te_preds), 1), 4),
        "seconds": round(time.time() - t0, 1),
        "device": str(device),
    }
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(a.out, "test_predictions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "pred"])
        for i, p in enumerate(te_preds):
            w.writerow([i, p])
    log("metrics: %s" % json.dumps(metrics, ensure_ascii=False), log_path)


if __name__ == "__main__":
    main()
