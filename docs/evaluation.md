# 评估与结果

results/summary.json 记录最新显式详情协议：SFT130/200、RL450138/200；逐题成功标记与步骤位于对应 per-task.jsonl。RL随机采样682/1000，至少一次成功174/200为独立覆盖指标。

原Reward v3严格成功要求完整done/over/purchase、gold_purchase、reward_valid=true且无基础设施无效。固定分母200，所有异常保留。两组同任务、贪心参数和35步上限；RL另外使用搜索串行化修复，差额不是完全隔离的RL效应。14671、2717需求不完整，保留且待复核。

完整评估代码另支持Flash整理冻结Rubric、Pro评价盲化轨迹，输出Reward/Rubric/Trajectory/Deterministic四个面板。当前65%/69%属于Reward面板，不意味着双Judge评分已完成。

离线示例入口只汇总已有示例，不运行模型：

    PYTHONPATH=src python -m shopping_grpo.cli evaluate examples/trajectories.jsonl

在线入口见 scripts/evaluate.sh 和 scripts/evaluate_shop_benchmark.py；模型服务及Judge凭据按参数准备。历史结果依赖冻结运行配置，本目录为整理后的实现，不声称新运行必然逐题相同。
