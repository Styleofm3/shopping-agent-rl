# 模型下载

两个模型已发布在 **Hugging Face / Styl4**。发布文件已逐项核对大小，以及 LFS SHA256 或 Git blob 哈希。

| 模型 | 下载入口 | 格式 / 大小 |
|---|---|---|
| 课程 SFT C | [Styl4/shopping-agent-rl-sft-2b](https://huggingface.co/Styl4/shopping-agent-rl-sft-2b) | Hugging Face 合并模型，约 4.45 GB |
| RL450 | [Styl4/shopping-agent-rl-rl450-2b](https://huggingface.co/Styl4/shopping-agent-rl-rl450-2b) | veRL/FSDP 四卡 actor，约 5.56 GB |

## 固定版本下载

安装 huggingface_hub 后，可按下面的发布版本下载；机器可读入口见 [model-downloads.json](model-downloads.json)。

```python
from huggingface_hub import snapshot_download

sft_dir = snapshot_download(
    "Styl4/shopping-agent-rl-sft-2b",
    revision="a0d32f6a38b67456774f74ae289a6e1882bbee8d",
)
rl_actor_dir = snapshot_download(
    "Styl4/shopping-agent-rl-rl450-2b",
    revision="6c84dbfd088a8f62328090d6a6b99e04ad851ba3",
)
```

每个模型仓库均提供模型卡、LICENSE 和 checksums.json。

## 加载格式与依赖

SFT C 包含完整合并权重、配置和 tokenizer，不需要额外的 SFT LoRA adapter。使用代码仓库锁定的 Transformers 版本及 Qwen3_5ForConditionalGeneration 加载；具体示例见模型卡。

RL450 保留四个 model_world_size_4_rank_*.pt 分片、FSDP 配置和 LoRA 元数据。huggingface 子目录仅有配置与 tokenizer，没有完整权重。恢复需要 SFT C 基座、FSDP v1 / world size 4 和匹配的 veRL actor 加载流程；本次没有验证独立转换或加载配方，不能将此仓库当作普通 Transformers 模型或独立 PEFT adapter。

发布清单不含 optimizer 与 extra_state，因此不是完整的训练续跑快照。本次未做模型合并、转换、训练或新评测。

## 结果与许可

现有 Final200 Clean 显式详情协议下，SFT C 为 130/200（65%），RL450 为 138/200（69%）。RL 评估还包含搜索串行化修复，4 个百分点不是完全隔离的 RL 效应。详见 [评估说明](evaluation.md)。

基础模型为 [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)。模型权重沿用 Apache-2.0，两个模型仓库均附其官方许可证文本；本项目代码采用 MIT。
