# 图表工程与实验协议

图表项目位于 [docs/experiments](../../../docs/experiments/README.md)，配置与轻量数值由 docs Git 维护；通用生成器位于 nerfstudio/research_artifacts。输入原图和历史重导出在 outputs/artifact-inputs，冻结证据包在 outputs/frozen-inputs，完整运行池在 outputs/eehpc。

从 paper1 根目录运行：

```bash
bin/paper1 artifacts --config docs/experiments/main-comparison/figure-2d-tmi.json
bin/paper1 artifacts --object table_joint_ablation
```

[artifacts.json](artifacts.json) 仅将已有对象 ID 路由到 docs 配置，不保存第二份配置。JSON 的 parameters 提供生成器支持的方法顺序、选帧裁剪、面板和配色等设置；不覆盖算法、训练预算或冻结数值。静态历史导出配置没有 generator，不冒充可执行入口。

每次生成写入 outputs/artifacts/<generator>/<build-id>，保存有效 artifact-config.json、源配置 SHA256、生成器 SHA256、读取数值和图片的哈希及 build-receipt.json。生成器只读输入，不导入旧训练器、不编译旧算子、不训练。

主实验由 [matrices/main.json](matrices/main.json) 和循环控制 624 个组合，`paper1 matrix` 默认只规划；正式 1M/30k 尚未开始。238 份原生 YAML 按需生成，原始科学配置仍在冻结证据包。旧证据通过 `paper1 history` 从保留的 Git 历史读取，不重新展开数万文件。

详细边界与历史路径解释见 [存储规约](../../../docs/protocols/artifact-storage.md)。outputs 中的原始渲染和冻结输入可能不可再生；被 Git 忽略不表示可随意删除。
