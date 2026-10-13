# 训练入口

主线为课程SFT→在线GRPO。所有命令从仓库根运行，先按README安装依赖并准备环境。

课程命令检查（不训练）：

    bash scripts/sft_curriculum.sh --dry-run

查看参数：

    python scripts/train_lora_sft.py --help
    python scripts/train_grpo.py --help

scripts/sft.sh 和通用实验调度默认使用 data/sft 下的当前C阶段划分，属于单阶段入口；三阶段训练使用 scripts/sft_curriculum.sh。不能将默认单阶段入口当成历史四卡课程实验。

训练配置见 configs/grpo.yaml。使用C阶段合并模型时显式传入 --model；硬件数通过 trainer.n_gpus_per_node 覆盖。--dry-run 会检查输入文件，因此需要模型和数据已就位。

历史RL采用组内4条轨迹、LoRA rank16/alpha32、学习率1e-6、关闭KL项。发布代码包含后续框架修订；已保存结果不是本发布目录新运行的结果。
