# main-comparison

本实验目录保存图表工程配置、来源和小型数值；执行代码在 Nerfstudio，原图和运行产物在 outputs。这里保留的是历史实验材料，不代表新代码正式复现。

## 四数据集八方法主结果表

- 配置：[table-main.json](table-main.json)
- 原始输入与历史导出：[outputs/artifact-inputs/table_main_comparison](../../../outputs/artifact-inputs/table_main_comparison)

```bash
bin/paper1 artifacts --config docs/experiments/main-comparison/table-main.json
```

## 二维重建对比图原始布局

- 配置：[figure-2d-original.json](figure-2d-original.json)
- 原始输入与历史导出：[outputs/artifact-inputs/figure_main_2d_original](../../../outputs/artifact-inputs/figure_main_2d_original)

```bash
bin/paper1 artifacts --config docs/experiments/main-comparison/figure-2d-original.json
```

## 四数据集八方法二维重建对比图 TMI 版本

- 配置：[figure-2d-tmi.json](figure-2d-tmi.json)
- 原始输入与历史导出：[outputs/artifact-inputs/figure_main_2d_tmi](../../../outputs/artifact-inputs/figure_main_2d_tmi)

```bash
bin/paper1 artifacts --config docs/experiments/main-comparison/figure-2d-tmi.json
```

命令从 paper1 根目录运行。生成内容写入 outputs/artifacts；每次构建保存 artifact-config.json、配置 SHA256、读取的源文件哈希与 build-receipt.json。

配置 parameters 控制生成器已支持的方法顺序、选帧与裁剪、面板或配色；不修改原始运行结果。assets/、exports/ 和 .cache/ 如在本目录生成，均被精确忽略。
