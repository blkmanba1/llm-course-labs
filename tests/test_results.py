# -*- coding: utf-8 -*-
"""结果一致性测试：results.csv 里写的数字必须与 labs/*/results/metrics.json 一致。

不需要 GPU，也不需要训练：只读文件。运行：
    python -m unittest discover -s tests -v
"""
import csv
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "results.csv")

# results.csv 的 metric 名称 -> 从对应 JSON 取值的函数
GETTERS = {
    "validation accuracy (seed 42)": ("seed42", lambda d: d["val_accuracy"] * 100, 0.01),
    "validation accuracy (seed 43)": ("seed43", lambda d: d["val_accuracy"] * 100, 0.01),
    "validation accuracy (seed 44)": ("seed44", lambda d: d["val_accuracy"] * 100, 0.01),
}


def read_csv():
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestResultsConsistency(unittest.TestCase):
    def test_every_source_exists(self):
        for row in read_csv():
            path = os.path.join(ROOT, row["source"].replace("/", os.sep))
            self.assertTrue(os.path.exists(path), "缺少结果文件：%s" % row["source"])

    def test_json_values_match_csv(self):
        """逐行核对：能把 CSV 的值映射回 JSON 的行，必须相等（百分比保留 2 位）。"""
        rows = read_csv()
        # ch1 三个种子
        for seed in ("seed42", "seed43", "seed44"):
            path = os.path.join(ROOT, "labs", "ch1_bert_disaster_tweets", "results", seed, "metrics.json")
            with open(path, encoding="utf-8") as f:
                acc = json.load(f)["val_accuracy"] * 100
            picked = [r for r in rows if r["source"].endswith("%s/metrics.json" % seed)]
            self.assertEqual(len(picked), 1, "%s 在 results.csv 里应恰好一行" % seed)
            self.assertAlmostEqual(float(picked[0]["value"]), round(acc, 2), places=2,
                                   msg="%s 的准确率与 CSV 不一致" % seed)

        # ch1 汇总
        path = os.path.join(ROOT, "labs", "ch1_bert_disaster_tweets", "results", "seeds_summary.json")
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        mean_row = [r for r in rows if r["metric"] == "mean validation accuracy"][0]
        sd_row = [r for r in rows if r["metric"] == "standard deviation"][0]
        self.assertAlmostEqual(float(mean_row["value"]), round(summary["mean"] * 100, 2), places=2)
        self.assertAlmostEqual(float(sd_row["value"]), round(summary["sd"] * 100, 2), places=2)

        # ch7
        path = os.path.join(ROOT, "labs", "ch7_gpt2_stegano", "results", "metrics.json")
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        for scheme, label in (("huffman", "Huffman"), ("flc", "fixed-length")):
            bpt = [r for r in rows if r["chapter"] == "ch7" and label in r["experiment"]
                   and r["metric"] == "bits embedded per token"][0]
            acc = [r for r in rows if r["chapter"] == "ch7" and label in r["experiment"]
                   and r["metric"] == "decode bit accuracy"][0]
            self.assertAlmostEqual(float(bpt["value"]), round(st[scheme]["bits_per_token"], 2), places=2)
            self.assertAlmostEqual(float(acc["value"]), round(st[scheme]["decode_bit_accuracy"] * 100, 2), places=2)

        # ch11
        path = os.path.join(ROOT, "labs", "ch11_rlhf_sentiment", "results", "metrics.json")
        with open(path, encoding="utf-8") as f:
            rl = json.load(f)
        checks = [
            ("mean reward (first 5 steps)", rl["mean_reward_first5_steps"]),
            ("mean reward (last 5 steps)", rl["mean_reward_last5_steps"]),
            ("mean reward (held-out prompts after training)", rl["mean_reward_eval_after"]),
            ("mean KL (k1 estimate)", rl["mean_kl"]),
        ]
        for metric, value in checks:
            row = [r for r in rows if r["metric"] == metric][0]
            self.assertAlmostEqual(float(row["value"]), round(value, 3), places=3,
                                   msg="%s 与 metrics.json 不一致" % metric)

    def test_no_missing_metric_rows(self):
        """results.csv 应覆盖三个实验，至少 13 行。"""
        rows = read_csv()
        self.assertGreaterEqual(len(rows), 20)
        chapters = {r["chapter"] for r in rows}
        self.assertEqual(chapters, {"ch1", "ch2", "ch6", "ch7", "ch10", "ch11"})


if __name__ == "__main__":
    unittest.main()
