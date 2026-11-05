# joint-ablation

当前依据 Mac manuscript-for-216-20261105-155440 文稿；完整逐图逐表核对见 [来源审计]（历史路径 `../manuscript-source-audit/README.md`，本轮已清理）。

- 配置：[table-joint.json](table-joint.json)

```bash
bin/paper1 artifacts --config docs/experiments/joint-ablation/table-joint.json
```

数值与配置原件在 outputs/paper-evidence，定性面板在 outputs/paper-images；不再保留整棵旧运行树。生成写入 outputs/paper-builds，原始图表 PDF 使用 docs/manuscript 的 Mac 副本，不以本机字体替换版本覆盖。

以上标注的历史路径不属于当前 Mac 文稿保留范围；当前图表来源以 manuscript-source-audit 的对应表和保留清单为准。
