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

[代码与运行入口](../nerfstudio/experiments/paper1/README.md)、[图表工程](experiments/README.md)、[新旧指标参照](experiments/native-5k-liver-l1/report.md)分别由对应仓库维护。数据正文只在 shared/datasets，历史权重不迁入。

旧代码文档和重复链接层已删除；图表工程与小型数值归入 docs/experiments，运行文件、重资产和批量导出归入 outputs。完整逐文件去向见 [整理记录](resources/maintenance-assets/source-inventories/docs-layout-20261105.json)，旧版本保留在本仓库 Git tag `history/docs-before-typed-layout-20261105`，无需在工作树保留归档镜像。

目录与配置分工见 [存储规约](protocols/artifact-storage.md)。独立 results 已取消，原 Git 历史单独保全；不保留兼容目录或链接。

当前论文金标准与保留范围见 [Mac 文稿逐图逐表审计](experiments/manuscript-source-audit/README.md)，完整 LaTeX 入口已收到并保存在 manuscript。旧迁移阶段“未找到完整文稿”的描述仅属于历史记录。
