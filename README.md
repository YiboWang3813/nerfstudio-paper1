# Paper1 研究文档

采用与 paper2 一致的用途分类。当前八方法使用原生 Nerfstudio / gsplat；已完成短程验收和代表 sweep 的 5k 指标参照，正式 1M/30k 尚未开始。历史报告保留原身份。

| 类别 | 内容 |
| --- | --- |
| [development](development/README.md) | 开发任务、实施计划与迁移整理 |
| [protocols](protocols/README.md) | 工作流程、方法公式、评估和图表规约 |
| [experiments](experiments/README.md) | 按研究对象组织的实验协议和执行记录 |
| [reports](reports/README.md) | 方法结果、诊断、数据检查和迁移验收 |
| [decisions](decisions/README.md) | 科学取舍与图形设计决策 |
| [resources](resources/README.md) | 数据集说明、论文和研究笔记 |
| [manuscript](manuscript/README.md) | 引言、方法、结果叙述和写作材料 |

[代码与运行入口](../nerfstudio/experiments/paper1/README.md)、[图表工程](../results/objects/README.md)、[新旧指标参照](../results/probes/native-5k-liver-l1/README.md)分别由对应仓库维护。数据正文只在 shared/datasets，历史权重不迁入。

旧代码文档和重复链接层已删除；图表导出与冻结数值输入已归入 results。完整逐文件去向见 [整理记录](../results/catalog/docs-layout-20261105.json)，旧版本保留在本仓库 Git tag `history/docs-before-typed-layout-20261105`，无需在工作树保留归档镜像。
