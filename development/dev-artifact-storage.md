# 实验记录、图表工程与运行产物归位

目的：取消独立 results 工作目录和日常仓库，将轻量研究记录与图表工程并入 docs，将全部 runs、日志、权重、原始渲染与生成输出归 outputs。共享数据、环境及科学协议不变。

计划与验收：

1. 按实验对象归入 docs/experiments；图表配置、来源、小型数值和报告跟踪 Git，原图和批量导出保留在 outputs 或被精确忽略的 assets/exports。
2. 通用生成器与训练/评估配置留在 Nerfstudio；图表工程配置由 docs 维护，运行保存实际有效配置快照与来源哈希。
3. 更新 artifacts、history 与探针汇总入口，取消 results 路径依赖；原始回执保持历史字节身份，通过迁移映射解释旧路径。
4. 保留 results Git 历史与固定证据 tag，移入 migration/git-history；不保留 results 兼容目录或软链接。
5. 验证资产去向、Git 忽略、链接、历史证据读取和受影响图表 CPU 重建；不启动训练，不改共享环境，不访问 R2。

已完成。源端文件不删除，历史 checkpoints 不迁移；现有本机权重原地保留在 outputs。

执行记录：16 个对象配置按研究目的归入 docs/experiments；保留通用生成器与单份对象路由表。历史运行池移至 outputs/eehpc，探针、重资产、冻结输入和批量导出分别归入 outputs/probes、artifact-inputs、frozen-inputs 与 artifacts。独立 results 目录及兼容链接均不存在，原 Git 数据库单独保全。

验收：全体 docs Markdown 链接有效，9 个冻结输入包 SHA256 一致；546 个直接迁移的旧跟踪资产逐字节匹配，其余原件、导航或已调整配置保留于原 results Git HEAD，33,399 文件证据树仍可按需读取。消融表、TMI 二维对比图和超参数图通过新配置入口重建；每次构建记录有效配置与哈希。八方法探针汇总 CPU 重生成通过，不重写原收据；精确 Git 忽略验证及 Ruff 通过。没有训练、R2 操作或共享环境变更。

原 results HEAD：200062a0a55e4284be36c8eb81fa351b49a895a2。最终代码版本：`da083c230fe453b33dfabc3b0a4cd33eb7dc1cfd`。详细记录见 [验收](../resources/provenance/artifact-storage-validation.json)。
