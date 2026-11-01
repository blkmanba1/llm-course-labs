# ch11 · 用情感奖励做 RLHF（最小策略梯度实现）

## 为什么不是原 notebook

课程原版用 `trl` 的 `PPOTrainer`（策略 `lvwerra/gpt2-imdb`、奖励 `lvwerra/distilbert-imdb`）。
本机环境是 `transformers 5.8` + `torch 2.11`，`trl` 的旧版 PPO 接口与之不兼容；
装新版 `trl` 又要重写数据管线与 `PPOConfig`，改动量超过本实验本身。

因此这里保留**课程的两个模型**，把训练循环重写为一个最小策略梯度（REINFORCE + KL 惩罚）：

```
loss = -(reward - baseline) * Σ logp_policy + kl_coef * Σ (logp_policy - logp_ref)
```

KL 用的是 k1 估计量（逐 token 对数概率差），不是 PPO 的 clip 目标，因此**这不是 PPO**，
只能说“用同一个奖励模型做了一次带 KL 约束的策略优化”。

## 设置

| 项 | 值 |
|---|---|
| 策略 / 参考 | `lvwerra/gpt2-imdb`（124M，参考模型冻结） |
| 奖励模型 | `lvwerra/distilbert-imdb`（67M，正面类概率） |
| 提示集 | 8 个中性开头（`This movie`、`The acting` 等），每步随机取 4 个 |
| 生成 | 20 token，top-k 50、top-p 0.95、采样 |
| 步数 / 学习率 | 60 步 / 1e-5（AdamW，梯度裁剪 1.0） |
| KL 系数 | 0.02 |

## 结果（`results/metrics.json`）

| 指标 | 值 |
|---|---|
| 前 5 步平均奖励 | **0.625** |
| 后 5 步平均奖励 | **0.979** |
| 训练后固定提示评估（8×4 条） | **0.899** |
| 平均 KL（k1 估计） | 1.17 |

## 一个必须写出来的问题：奖励被“刷”了

训练后的样例（`results/samples.txt`）几乎都是同一副面孔：

- `This movie is a very good thriller from start to finish...`
- `The acting was good. There was a great soundtrack that was perfect...`

句子通顺，但**内容空洞、形容词堆叠**——这是典型的 reward hacking：策略发现“把情感词写满”就能拿到高分，
而奖励模型只看情感极性，看不出这种退化。这也是为什么真实 RLHF 需要
①更细的奖励模型（或人评）、②对 KL 的严格约束、③留出独立评审。
本实验的 KL 系数只有 0.02，60 步内 KL 已经升到 1.17，说明约束偏松。

## 复现

```powershell
D:\AAA_my_study\大模型\llm_env\Scripts\python.exe run_rlhf.py
```

约 40 秒（8 GB 卡）。模型从 `https://huggingface.co` 拉取（`hf-mirror.com` 对这两个仓库返回元数据错误）。
