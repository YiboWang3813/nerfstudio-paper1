# 实验记录、图表工程与运行产物

代码与训练/评价配置维护在 nerfstudio/gsplat，图表 JSON、报告和小型数值维护在 docs/experiments。数据正文仅在 shared/datasets，运行日志、权重、预测、渲染和生成文件不进源码或文档 Git。

当前保留范围以 Mac manuscript-for-216-20261105-155440 的五图五表为准，逐项来源见 [文稿审计](../experiments/manuscript-source-audit/README.md)。正式 1M/30k 未启动。历史结果和本机重建图表不能冒称新代码正式实验。

| 位置 | 当前内容 |
| --- | --- |
| outputs/paper-evidence | 270 个图表输入/构建审计依赖，以及 774 个相关 run 的 5155 个原始数值与配置文件；按 ZIP 保存并逐项验证 SHA256 |
| outputs/paper-images | 正式定性图实际使用的 36 个面板 |
| outputs/paper-figure-sources | Mac 对应 endo 图的可编辑 SVG、数值与生成来源记录 |
| outputs/paper-builds | 仅保留当前论文对应的最近一次验证构建；后续构建保存有效配置、源配置和输入哈希 |
| docs/manuscript | Mac 完整 TeX 源项目，作者 PDF 仅本机保留并忽略，build/dist 不跟踪 |

从 paper1 根目录运行 `bin/paper1 artifacts --config docs/experiments/main-comparison/figure-2d-tmi.json`。对象 ID 由代码仓库 artifacts.json 路由至同一份 docs 配置，不复制配置。图表生成不会训练。

`bin/paper1 history <原始outputs相对路径>` 按需读取保留的 run ZIP；没有纳入 Mac 文稿来源链的旧文件不再提供。旧 results Git 数据库和 migration 杂物目录已退出；原始收据不追改，来源哈希与历史执行提交仍保留为事实。

重文献、本地重素材与详细维护日志按明确路径忽略；不得忽略整个实验目录或全局忽略所有图像而隐藏复现配置和来源。新的训练若日后获授权，仍按 outputs/<study>/<runID> 保存运行文件，不能擅自复用旧论文原件目录。
