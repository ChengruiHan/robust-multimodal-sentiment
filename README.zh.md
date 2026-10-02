# 鲁棒多模态情感识别

本项目研究文本、语音和视觉观测不完整时的情感预测，以及冻结模型的证据分析。
**STATE-MSA** 输出负向、中性、正向三类概率与连续情感强度；**TRACE-MSA**
提供模态 Shapley 归因和局部窗口删除分析；视频特征抽取模块保留词级对齐、
可用性与来源信息。[English overview](README.md)。

公开版包含代码与文档。原始数据、媒体、数值预测、解释记录、模型权重与 scaler
均需另行准备。

## 阅读路径

参考 [OptiCall](https://github.com/ChengruiHan/OptiCall)，项目按“方法 → 实验流程 →
结果说明 → CPU 检查 → 完整复现”组织。方法见 [method](docs/method.md)，
数据契约见 [data contracts](docs/data-contracts.md)，复现命令见
[REPRODUCE](docs/REPRODUCE.md)，实验演变见 [HISTORY](docs/HISTORY.md)。

| 模块 | 作用 | 入口 |
| --- | --- | --- |
| STATE-MSA | 带可用性掩码的情感分类与条件强度回归 | [sentiment_model](sentiment_model/README.md) |
| TRACE-MSA | 冻结模型的模态归因、局部证据和近似时间映射 | [explainability](explainability/README.md) |
| 特征抽取 | 视频与文本的词级特征、对齐和质量审核 | [feature_extraction](feature_extraction/README.md) |

预测模型使用 50 个对齐位置和 74/35 维声画特征；视频抽取输出逐词的 50/40 维
声画特征，二者不能直接串联。完整组件环境使用 Python 3.12，两个模块的依赖
锁文件分别保留。

## 已记录结果

| 验证条件 | 每条件样本数 | Accuracy | Macro-F1 | MAE |
| --- | ---: | ---: | ---: | ---: |
| 完整输入 | 728 | 0.6429 | 0.6279 | 0.5660 |
| 54 个受控局部缺失条件的均值 | 728 | 0.6135 | 0.5981 | 0.5917 |

数值沿用原项目的验证记录，公开仓库没有提供逐样本预测或重新推理的资产。
54 指的是六种模态组合、三个缺失比例和三个连续块位置形成的条件数量，每个
条件使用相同的 728 条验证样本。验证集参与过开发，不能视为盲测。
[评估说明](docs/evaluation.md) 记录了这些边界。

## 无需 GPU 的快速检查

在仓库根目录使用 Python 3.11 或 3.12，只需标准库：

```bash
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

测试使用人工构造的数值和临时文件，不需要研究数据。模态归因、特征聚合和模型
掩码的合成测试需 NumPy 或 PyTorch，环境和命令见 [安装](docs/INSTALL.md) 与
[复现指南](docs/REPRODUCE.md)。

额外提供私有数值记录的哈希验证和指标复算工具，需要显式指定外部目录；文件
格式见 [DATA](docs/DATA.md)。完整训练、冻结推理、解释和抽取同样需要外部资产。
仓库当前没有许可证，复用前请联系作者。
