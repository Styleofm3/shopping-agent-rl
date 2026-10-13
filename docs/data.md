# 数据与资源

本仓库保留当前训练和评估资源，不包含历史实验目录。

| 目录 | 内容 |
|---|---|
| data/sft_pure_v4 | DeepSeek V4 Flash 教师轨迹，1192唯一任务；标签和去重记录 |
| data/sft_curriculum | A/B/C累计课程，固定训练/验证划分 |
| data/sft | 从当前课程导出的C阶段1073/119清单对应轨迹，供通用单阶段入口使用 |
| data/grpo | 1000训练任务、50验证任务及Parquet |
| data/evaluation | 固定200题清单与元数据 |
| data/catalog | 历史评估修复版商品压缩包及公开规格证据 |

C阶段实际预处理后的1069/118样本数属于历史训练记录，不能当作原始清单数。SFT/RL与Final200的任务隔离由发布检查核验；SFT与RL并非完全不相交，交集另见数据审计。

普通商品包位于 environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz，setup.sh 校验解压后建立索引。修复版使用 data/catalog/reviewed-products.json.gz；两套数据不要混配索引。商品和任务私有字段属于环境/评估数据，不应拼入Actor提示。

模型权重通过独立渠道发布，见 models.md。
