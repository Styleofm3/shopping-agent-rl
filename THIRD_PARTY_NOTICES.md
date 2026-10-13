# 许可与第三方资源说明

本项目代码采用 [MIT License](LICENSE)。第三方代码、模型和数据保留各自适用的许可、版权和来源声明；项目 MIT 许可不替代这些条款。

| 组成 | 许可与来源 |
|---|---|
| 项目代码 | [MIT License](LICENSE) |
| Qwen3.5-2B 基础模型 | Apache-2.0，见 [官方模型许可证](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE) |
| ShopSimulator 环境 | 见 [项目仓库](https://github.com/ShopAgent-Team/ShopSimulator)；版本与来源记录保存在 environments/ShopSimulator/EMBEDDED_SOURCE.json，适用原组件条款 |
| 商品数据及修复版 | 来源与处理记录见环境资源和 data/catalog/metadata.json；第三方商品内容保留原有权利 |
| 教师轨迹 | 使用 DeepSeek V4 Flash 生成，数据记录见 data/sft_pure_v4/metadata.json；适用的服务条款和第三方内容权利仍予保留 |

PyTorch、Transformers、PEFT、veRL、vLLM 等通过依赖管理器安装，具体许可以所安装版本随附的许可证和通知为准。

[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) 提供 Apache-2.0 标准全文，供采用该许可的组件使用。模型权重发布时保留对应版本随附的许可证与通知。
