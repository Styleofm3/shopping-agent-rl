# 模型下载

模型计划托管在 **Hugging Face**。当前尚未创建账号或模型仓库，以下为占位，暂无可下载权重。

| 模型 | 拟用仓库名 | 下载状态 | 格式 / 大小 |
|---|---|---|---|
| 课程 SFT C | shopping-agent-rl-sft-2b | 待发布 | Hugging Face 合并模型，约4.45 GB |
| RL450 | shopping-agent-rl-rl450-2b | 待发布 | veRL/FSDP四卡 actor，约5.56 GB |

账号确定后，仓库ID为“用户名/拟用仓库名”。[机器可读入口](model-downloads.json)中的 repo_id 和 download_url 暂为 null；不是失效链接。

## 基础模型

基础模型 [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) 已公开，但不是本项目训练后的权重。其官方许可为 [Apache-2.0](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE)。

## 发布格式

SFT C 已有合并模型与tokenizer。RL450 仍为四个 model_world_size_4_rank_*.pt 分片；huggingface 子目录只有配置与tokenizer，不可单独当作完整模型加载。RL产物还需要C阶段基座及匹配的加载流程，正式模型卡会明确该依赖。

本轮未合并、导出或上传模型。发布清单排除 optimizer 与 extra_state；发布前按文件记录哈希，补齐加载说明及许可信息。

账号准备好后，依次创建仓库、上传所需权重、校验文件、填写真实repo_id与下载URL，最后更新本页。未完成验证前仍保留“待发布”状态。
