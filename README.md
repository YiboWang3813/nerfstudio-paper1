# Paper1 研究文档

当前代码和运行协议以 [Nerfstudio](../nerfstudio/experiments/paper1/README.md) 为准。本目录保存研究说明、文稿和历史证据；历史结果尚未由原生重构版本复现。

| 入口 | 用途 |
| --- | --- |
| [当前研究导航](current/README.md) | 八种保留方法、四类数据、当前验证与文稿边界 |
| [历史论文证据](archive/eehpc-20261017/04_experiments/production/final_paper_evidence_20260910/README.md) | 源端最终证据包，保留原始实验身份 |
| [历史消融表](archive/eehpc-20261017/04_experiments/production/final_ablation_tables_20260916/README.md) | 原始消融表及来源 |
| [历史文稿](archive/eehpc-20261017/05_manuscript/) | 写作材料与旧图表；不作为新代码结果 |
| [文献](archive/eehpc-20261017/06_literature/) | 已有研究来源与 PDF |
| [文件来源清单](catalog/source.json) | 来源提交、数量与迁移范围 |
| [逐文件校验](catalog/files.jsonl) | 原路径、新路径、Git blob、SHA256 |

源端版本化内容按原目录结构冻结于 `archive/eehpc-20261017/`，内部相对引用保持原样。跨仓库的旧绝对路径、旧环境和 `../protocol` 等引用仅描述历史工作区，不能直接作为本机运行命令。重复内容记录在 `catalog/duplicates.json`，未删除历史证据包中的重复配置。

数据正文在 `/home/yibo/shared/datasets`；历史 checkpoints 尚未迁移。当前没有启动正式 1M/30k 实验。
