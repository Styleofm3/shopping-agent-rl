# 增强购物执行框架

A核验提示、B购买检查、C接口反馈、D观察预算、E上下文管理、F商品数据与索引、G购买与候选判定可分别配置。默认普通入口不会自动切换到增强适配器。

scripts/build_component_profile.py 检查商品/索引指纹并生成 agent_loop.yaml、tools.json、service.json 和 protocol.json。需要提供解压后的修复商品与由它生成的索引；可用 --disable 逐项关闭组件。示例：

    python scripts/build_component_profile.py --products data/catalog/reviewed-products.json --index outputs/reviewed-index.sqlite3 --output outputs/component-profile

该命令只生成配置，不启动模型或环境。service.json 的 argv/env 指明增强环境启动要求，GRPO启动时通过 --agent-config/--tool-config 选择生成文件。公开规格证据通过 serve_reviewed_environment.py 的 --option-evidence 显式指定。

商品解压与索引构建参数以对应脚本 --help 为准，元数据中的SHA256用于核对版本。购买检查只核对已实现的公开条件，不代表检查所有语义需求。商品详情保持显式工具获取。

组件消融及搜索修复差异见README；不能相加各项移除差值来推算组合收益。
