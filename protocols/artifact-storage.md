# 实验记录、图表工程与运行产物

| 位置 | 内容 | 版本管理 |
| --- | --- | --- |
| nerfstudio / gsplat | 算法、训练与评估实现、方法默认配置、参数矩阵、通用生成器 | 对应代码 Git |
| docs/experiments/<study> | 图表配置、来源、报告、小型数值与交换表格 | docs Git |
| outputs/<实验或运行目录> | 所有 runs 的权重、日志、有效配置、逐帧指标、渲染、视频、点云及批量导出 | 不进代码或 docs Git |
| shared/datasets | 数据正文 | 遵守冻结版本与发布政策 |

训练和评价的可执行配置由 Nerfstudio 维护，实验矩阵通过代码循环展开。图表工程配置由 docs 维护，规定输入来源、方法顺序、指标、面板、选帧和裁剪等生成器支持的参数；算法与数值计算逻辑只维护在代码仓库。实际执行保存有效配置快照、代码/脚本身份和输入校验。

示例：从 paper1 根目录运行 `bin/paper1 artifacts --config docs/experiments/main-comparison/figure-2d-tmi.json`。兼容的对象 ID 命令通过代码仓库 artifacts.json 路由到同一份 docs 配置，不复制配置。

每次构建写入 outputs/artifacts/<generator>/<build-id>，保留 artifact-config.json 和 build-receipt.json。配置源 SHA256 与实际读取的图片哈希用于追溯；不会启动训练。原始冻结证据包在 outputs/frozen-inputs，其 SHA256 清单在 [evidence-packages.json](../resources/provenance/evidence-packages.json)。

历史运行池位于 outputs/eehpc；探针实际文件位于 outputs/probes，训练输出仍在 outputs/results-probes。保留的图表原图和历史重导出在 outputs/artifact-inputs；这些可能不可再生，不能当缓存随意删除。小型数值与报告放回各实验目录。

确需随工程保存的人工素材可放实验目录 assets/，批量导出放 exports/，这些路径精确忽略。不能忽略整个实验目录或全局忽略全部 PNG/PDF，避免吞掉配置、来源与有意选择的文档插图。

独立 results 已取消。原 Git 数据库及其 tag 保存在 migration/git-history/results.git；`bin/paper1 history <相对证据路径>` 按需读取原字节。原配置、数据版本、收据、时间与执行身份不追改，旧路径由 [迁移映射](../resources/provenance/artifact-storage-map.json) 解释。
