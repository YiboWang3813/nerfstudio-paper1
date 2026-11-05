# plgr-sensitivity

本实验目录保存图表工程配置、来源和小型数值；执行代码在 Nerfstudio，原图和运行产物在 outputs。这里保留的是历史实验材料，不代表新代码正式复现。

## H2 PLGR 超参数图

- 配置：[figure-h2.json](figure-h2.json)
- 原始输入与历史导出：[outputs/artifact-inputs/figure_h2_plgr](../../../outputs/artifact-inputs/figure_h2_plgr)

```bash
bin/paper1 artifacts --config docs/experiments/plgr-sensitivity/figure-h2.json
```

命令从 paper1 根目录运行。生成内容写入 outputs/artifacts；每次构建保存 artifact-config.json、配置 SHA256、读取的源文件哈希与 build-receipt.json。

配置 parameters 控制生成器已支持的方法顺序、选帧与裁剪、面板或配色；不修改原始运行结果。assets/、exports/ 和 .cache/ 如在本目录生成，均被精确忽略。
