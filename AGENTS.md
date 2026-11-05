# Paper1 文档维护

继承 `/home/yibo/AGENTS.md` 与 `/home/yibo/paper1/AGENTS.md`。

- 沿用 paper2 的用途分类：development、protocols、experiments、reports、decisions、resources；paper1 另有 manuscript。
- development 保存开发任务与实施计划，文件名 `dev-<task>.md`；protocols 保存稳定规约、公式、评估与图表约定；reports 保存事实、结果与排查结论；decisions 保存有理由的取舍。
- experiments 仅用一层清晰的实验 ID，内部使用 protocol、report、status 等有意义文件名。普通文档直接放类别根目录；文献统一在 resources/papers，研究笔记包在 resources/research-notes。
- 文件名使用英文小写和短横线，不用数字目录、日期层层套娃、archive、supplement、source_workspace 或重复目录软链接。不按原服务器目录重新镜像。
- 同类规约集中在 protocols，内容更新时明确版本适用范围；父级 AGENTS 是本机资源和环境规则的权威来源。历史 Slurm/Hydra 命令不得冒充本机入口。
- 算法、训练/评估配置和通用生成器属于 nerfstudio/gsplat；图表项目配置、来源、小型数值与报告属于 docs/experiments/<study>。所有 runs、权重、日志、原图和批量导出属于 outputs。禁止在文稿复制 source_snapshot，禁止重建独立 results。
- 历史研究文档保留历史身份。原生接入测试、5k 探针和历史 30k 结果不能写成新代码正式 1M/30k 已复现。
- 历史迁移去向见 resources/provenance；当前保留范围以 experiments/manuscript-source-audit 为准。旧代码与大文件已清理，不为追溯恢复旧树。

- 图表配置用 JSON，支持 bin/paper1 artifacts --config docs/experiments/<study>/<figure-or-table>.json；训练矩阵仍在代码仓库。实际执行配置快照保存在 outputs。参见 protocols/artifact-storage.md；assets/exports 的忽略规则不能吞掉配置、数值或来源。

- Mac 文稿已由 exchange 批次收到；五图五表是当前保留金标准。重文献正文、作者 PDF、维护大清单精确忽略。旧 results Git 数据库不再保留，原 run 文件通过 paper-evidence.json 指向的验证 ZIP 读取。
