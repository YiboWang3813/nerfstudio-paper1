# Paper1 跨仓库协议与资产索引

此目录独立于模型代码仓库。当前本地仓库的远端保存位置为
`YiboWang3813/nerfstudio-paper1` 的 `paper1/protocol` 分支。

- `assets/data.jsonl`：外部数据文件 SHA256、大小、原修改时间和 NPY 头。
- `assets/checkpoints.jsonl`：外部检查点目录内文件的内容身份与大小。
- `assets/derived_artifacts.jsonl`：派生数组、帧索引和源码/二进制冻结产物的内容身份。
- `releases/`：历史正式结果和图表来源的当前位置索引，保留原执行身份。
- `migration/`：原路径映射、按修改时间排序的导入记录及验收摘要。
- `tools/`：迁移、校验与旧实验对照工具。
- `slurm/`：通过计算节点执行的功能对照任务。
- `workspace.lock.json`：代码、文档、结果分支及运行环境的精确版本。

迁移后的算法源码保持原样；唯一公开源文件清理是 legacy README 中的疑似凭据备忘。
原件、原 Git 历史和工作树补丁存放在 `../results/migration/2026-10-17/private/`，
该目录不推送到公开远端。修改时间只用于排序和来源记录，不伪造 Git 作者时间。

历史配置继续使用原 Hydra alias；新框架的原生 Nerfstudio Pipeline 适配尚未替代
这些实验。修改算法或环境后必须另建 run ID，并完成与已保存 reference 的对照。

从工作区运行文件完整性验证：

```bash
python protocol/tools/verify_migration.py
```

GPU 功能对照在原 `gaussian_splatting` 环境执行：

```bash
sbatch protocol/slurm/verify_migration.sh
```

复跑验证脚本前应换用新的验收输出目录，避免覆盖上次证据。
