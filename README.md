# Paper1 实验结果与证据

- `legacy/outputs/`：原运行产物，完整保留在外部磁盘目录。
- `evidence/legacy/`：从运行产物提取的数值表、配置、manifest 和报告快照，纳入 Git。
- `releases/`：必要的图表输入和迁移验收摘要，纳入 Git。
- `migration/`：搬迁日志、源 Git 历史、私有原件和测试运行产物，本地保留。
- `runs/`：后续新运行输出。

此目录的独立 Git 仓库推送到 Nerfstudio fork 的 `paper1/results` 分支。
数据和权重分别位于 `../data/` 与 `../checkpoints/`；代码仓库不承载其内容。
数值证据快照不能替代完整数据/模型的持久备份。
