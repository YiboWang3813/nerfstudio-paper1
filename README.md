# Paper1 研究文档

研究文档物理独立于 Nerfstudio、gsplat 代码目录，并由独立 Git 工作树保存。
远端是 `YiboWang3813/nerfstudio-paper1` 的 `paper1/docs` 分支。
移出代码目录不等于取消版本管理，所有迁入文档和已有图表证据都需要跟踪。

| 目录 | 内容 |
| --- | --- |
| `00_index/` | 导航、旧路径映射和原始修改时间顺序 |
| `01_problem_and_methods/` | 方法、数学、创新方向和自实现比较方法 |
| `02_datasets/` | 数据来源、物理几何、预处理、QC 与划分说明 |
| `03_engineering/` | 当前工作区规约与历史工程规约 |
| `04_experiments/` | 实验协议、正式结果、审计及已有证据包 |
| `05_manuscript/` | 论文写作交接材料 |
| `06_literature/` | 文献与研究来源 |
| `90_archive/` | 被替代的索引和历史材料 |
| `audits/` | 迁移前审计 |

`00_index/legacy_path_map.json` 按原文件修改时间升序排列，记录全部原文件的
新旧路径、mtime、大小和 SHA256。文件内容没有因分类而改写。
已有自包含证据包保留内部目录，防止拆散数值表、图片和来源 manifest。
当前规约以 `03_engineering/workspace_rules.md` 为准，历史工程文档是冻结记录。

数据实际在 `../data/`，检查点在 `../checkpoints/`，旧实验产物在
`../results/legacy/outputs/`。数据与结果版本身份见 `../protocol/`；研究说明
不再充当新实验的运行配置。历史脚本访问 docs 下矩阵的兼容接口暂时保留。

最终论文证据入口：

- [最终证据包](04_experiments/production/final_paper_evidence_20260910/README.md)
- [最终消融表](04_experiments/production/final_ablation_tables_20260916/README.md)
- [超参数图的图形约定](04_experiments/production/final_hyperparameter_figures_20260916/FIGURE_CONTRACT.md)
- [工作区规约](03_engineering/workspace_rules.md)

历史执行 SHA、配置和数值不会因搬迁被改写为新 fork 的执行结果。
