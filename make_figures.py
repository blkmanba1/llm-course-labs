# -*- coding: utf-8 -*-
"""从 results 生成两张图到 docs/figures/。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "figures")
os.makedirs(OUT, exist_ok=True)


def fig_ch1():
    p = os.path.join(HERE, "labs", "ch1_bert_disaster_tweets", "results", "seeds_summary.json")
    if not os.path.exists(p):
        print("skip ch1: no seeds_summary.json")
        return
    d = json.load(open(p, encoding="utf-8"))
    seeds = [str(s) for s in d["seeds"]]
    accs = [a * 100 for a in d["val_accuracy"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=160)
    bars = ax.bar(seeds, accs, color="#4C6EF5", width=0.55)
    ax.axhline(d["mean"] * 100, color="#E8590C", ls="--", lw=1.2,
               label="mean %.2f%% (sd %.2f)" % (d["mean"] * 100, d["sd"] * 100))
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.12, "%.2f%%" % a, ha="center", fontsize=9)
    ax.set_ylim(min(accs) - 2.5, max(accs) + 1.2)
    ax.set_xlabel("random seed")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("ch1 · BERT disaster-tweet classifier (1 epoch, effective batch 32)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch1_val_accuracy.png"))
    print("wrote ch1 figure")


def fig_ch7():
    p = os.path.join(HERE, "labs", "ch7_gpt2_stegano", "results", "metrics.json")
    if not os.path.exists(p):
        print("skip ch7: no metrics.json")
        return
    d = json.load(open(p, encoding="utf-8"))
    names = ["Huffman", "Fixed-length"]
    keys = ["huffman", "flc"]
    bits = [d[k]["bits_per_token"] for k in keys]
    sent = [d[k]["bits_sent"] for k in keys]
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=160)
    bars = ax.bar(names, bits, color=["#4C6EF5", "#12B886"], width=0.5)
    for b, v, s in zip(bars, bits, sent):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, "%.2f bits/token\n%d bits, 100%% decoded" % (v, s),
                ha="center", fontsize=8)
    ax.set_ylim(0, max(bits) + 0.9)
    ax.set_ylabel("bits embedded per generated token")
    ax.set_title("ch7 · GPT-2 text steganography (200 generated tokens)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch7_bits_per_token.png"))
    print("wrote ch7 figure")


if __name__ == "__main__":
    fig_ch1()
    fig_ch7()
