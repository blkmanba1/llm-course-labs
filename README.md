# 大模型课程实验复现（Dive into LLMs · 个人版）

本仓库是我对《动手学大模型》（`Lordog/dive-into-llms`，11 章）的**个人复现记录**：把我实际跑通的章节整理成可重复运行的脚本、日志与指标。课程原始 notebook 与课件仍在上游仓库，本仓库只放我自己的代码、运行记录和结果，便于复核，不替代原课程材料。

## 环境（实测）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti，8 GB 显存 |
| 驱动 / CUDA | 610.88 / torch 2.11.0+cu128 |
| Python | 3.11.3（venv：`D:\AAA_my_study\大模型\llm_env`） |
| 关键库 | transformers 5.8.0、datasets 4.8.5、pandas 3.0.3、scikit-learn 1.8.0 |
| 磁盘 | 约 105 GB 可用 |

## 结果表

| 章 | 实验 | 结果 | 本机耗时 | 产物 |
|---|---|---|---|---|
| ch1 | BERT 微调：灾害推文二分类（6793 行训练数据中的 6090 行、1523 行验证） | 验证集准确率 **83.19 / 83.98 / 83.78 %**（seed 42/43/44），均值 **83.65 %**，标准差 0.41 | 约 5 分钟/次（8 GB 卡，1 epoch） | `labs/ch1_bert_disaster_tweets/results/`、`docs/figures/ch1_val_accuracy.png` |
| ch7 | GPT-2 文本隐写：霍夫曼编码 vs 定长编码 | 200 token 内嵌入 **345 / 400 比特**（**1.73 / 2.00 比特/token**），补写解码器后比特恢复率 **100%** | 约 2 秒/次（CPU 也可跑） | `labs/ch7_gpt2_stegano/results/`、`docs/figures/ch7_bits_per_token.png` |

![ch1](docs/figures/ch1_val_accuracy.png)
![ch7](docs/figures/ch7_bits_per_token.png)

未运行的章节与原因见 [`not_run.md`](not_run.md)（大显存或 API key 依赖）。

## 目录

```
labs/
  ch1_bert_disaster_tweets/   BERT 微调：灾害推文二分类（课程第 1 章）
    train_bert.py             训练与评测脚本（支持 --seed）
    run_seeds.py              多随机种子重复，输出 mean±sd
    results/                  日志、指标、测试集预测
  ch7_gpt2_stegano/           GPT-2 文本隐写（课程第 7 章）
    run_stegano.py            霍夫曼编码 / 固定长度编码两种实现
    results/                  生成文本、比特嵌入统计
not_run.md                    未运行章节与卡点（显存 / API key）
```

## 复现步骤

```powershell
# 1) 环境
D:\AAA_my_study\大模型\llm_env\Scripts\python.exe -m pip install -r requirements.txt

# 2) 第 1 章：BERT 微调（约 4–5 分钟/次）
cd labs\ch1_bert_disaster_tweets
D:\AAA_my_study\大模型\llm_env\Scripts\python.exe train_bert.py --epochs 1 --batch 8 --accum 4 --max_len 512 --seed 42
D:\AAA_my_study\大模型\llm_env\Scripts\python.exe run_seeds.py        # 42/43/44 三个种子

# 3) 第 7 章：GPT-2 隐写（约 10 秒）
cd ..\ch7_gpt2_stegano
D:\AAA_my_study\大模型\llm_env\Scripts\python.exe run_stegano.py
```

数据路径可用环境变量覆盖：`CH1_DATA=<chapter1/TextClassification/TextClassification/data>`。
模型下载走 `HF_ENDPOINT=https://hf-mirror.com`（脚本已默认设置）。

## 与课程原版的差异

1. **第 1 章**：上游用 `run_classification.py` + `transformers.Trainer`，在本机 `transformers 5.8` 下不兼容；我改用等价的显式训练循环（AdamW + 线性预热衰减 + 梯度累积），超参对齐上游 `train.sh`（1 epoch、lr 2e-5、有效 batch 32、max_len 512）。测试集在原数据中无标签，故只报告验证集指标，测试集仅输出预测。
2. **第 7 章**：notebook 的生成上限 30 步，只嵌入约 56 比特；脚本把上限提到 200 步，并额外记录「嵌入比特数 / 生成 token 数」，用于比较两种编码的容量。

## 许可与归属

课程 notebook、课件与数据归原作者；本仓库仅包含我的运行脚本、日志与指标，供个人申请材料与面试复核使用。若上游仓库有具体许可条款，以该条款为准。
