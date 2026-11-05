# Paper1 图表工程与实验协议

当前图表对象入口在 [results/objects](../../../results/objects/README.md)。每个对象区分输入、协议、脚本、历史导出和可重建的本地输出；原始实验身份不因迁移而改写。

使用本项目入口，例如：

```bash
/home/yibo/paper1/bin/paper1 artifacts --object figure_main_2d_tmi
/home/yibo/paper1/bin/paper1 artifacts --object table_joint_ablation
```

图表代码在 `nerfstudio/research_artifacts`，仅读取本机冻结证据，不导入旧训练器或旧算子。输出写入新的目录。二维图对象还单独保留实际使用的 PNG、逐帧指标和汇总表，重建不依赖整棵旧运行输出目录。

238 份最终论文冻结协议已转换为 [原生配置](native-protocols/index.json)，包括 canonical uniform、file_step_norm 和 file_axis_projection。它们通过原生 YAML 读回；四份非均匀协议通过真实数据解析验证。配置准备不会启动训练。

训练使用原生 Nerfstudio Trainer/DataParser/DataManager/Model/Pipeline 及 gsplat 策略和算子。历史训练代码仅保存在 `source_reference` 供追溯，不能当作当前运行入口。历史 checkpoints、旧环境和编译二进制未迁入。

完整旧运行输入保留在本机结果归档，SHA256 和消费者关系分块记录在 `results/catalog/complete-assets-20261105`。被 Git 忽略的源归档仍可能包含不可再生的历史渲染，不能按普通缓存清理；只有对象 `builds` 和 `artifact-builds` 是可重建输出。

## 循环与按需生成

主实验使用 `matrices/main.json`，通过 `paper1 matrix` 查看 624 个组合；显式指定 `--train` 才按循环逐个运行。没有提前展开 624 份 YAML。旧证据通过 `paper1 history` 按需读取，完整来源保存在 results/history.json 指定的 Git 快照。238 份原生协议 YAML 也改为按需生成；冻结原始科学协议仍保留。
