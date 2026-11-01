# 未运行 / 运行受限的章节

记录时间：2026-11-01。硬件：单卡 RTX 4060 Ti，8 GB 显存；磁盘剩余约 105 GB；无付费 API key 时可跑的章节从略。

| 章 | 内容 | 卡点 | 结论 |
|---|---|---|---|
| ch2 | 提示学习与思维链 | notebook 走 OpenAI 兼容接口，需要 API key | 未跑（需要 key） |
| ch3 | 知识编辑（EasyEdit） | EasyEdit 典型配置需 16 GB 以上显存，且要下载 6–7B 权重 | 未跑（显存不足） |
| ch4 | 数学推理 SFT | 课程 README 明确要求 ≥40 GB 显存 | 未跑（显存不足） |
| ch5 | 文本水印（Baichuan-7B + OpenAI key） | 7B 权重 + API key 双重依赖，8 GB 需 4-bit 量化 | 未跑（显存与 key 均不足） |
| ch6 | 越狱攻击 | notebook 需要 api_key | 未跑（需要 key） |
| ch8 | 多模态大模型 | 需要 InternVL 级视觉语言模型权重与显存 | 未跑（显存不足） |
| ch9 | GUI 智能体（Qwen2-VL-7B + LLaMA-Factory） | 课程 README 写明至少 3×80 GB A100 | 未跑（显存不足） |
| ch10 | 智能体（OpenAI 接口） | 需要 OPENAI_API_KEY | 未跑（需要 key） |
| ch11 | RLHF 安全对齐 | 需要多卡与大显存，训练时长按小时计 | 未跑（显存不足） |

处理原则：这些章节不写进个人陈述；等有了云 GPU 或 API key 再补跑，届时把真实指标补到 `README.md` 的结果表与对应 `labs/` 目录。
