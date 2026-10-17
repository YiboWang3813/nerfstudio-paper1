# Paper1 存储迁移与旧实验兼容验收

本阶段完成资产搬迁、旧运行框架和 CUDA 后端的版本保存、研究文档分类、
数值证据快照以及旧实验兼容验收。原生 Nerfstudio Model/Pipeline 适配、
gsplat 统一 CUDA 构建与现代 ABI 移植属于下一阶段，尚未宣称完成。

## 实际目录

| 目录 | 内容与状态 |
| --- | --- |
| `../nerfstudio/paper1_legacy/` | 原框架、方法、Hydra 配置、脚本与测试；算法文件保持原字节 |
| `../gsplat/ultrasound_ops/` | 原自实现 CUDA 后端与共享头文件 |
| `../data/` | 原处理数据，593 个文件、5,888,686,086 字节；逐文件 SHA256 已保存 |
| `../checkpoints/legacy/` | 1,917 个原检查点目录及独立权重的外置存储 |
| `../checkpoints/pretrained/` | 四份指标预训练权重的独立副本，SHA256 与原缓存一致 |
| `../results/legacy/outputs/` | 原实验输出，原目录结构与内容保留 |
| `../results/evidence/legacy/` | 配置、manifest、scalar/per-frame 数值表和报告的 Git 快照 |
| `../results/releases/` | 组合图的 36 张原始输入和迁移验收摘要 |
| `../cache/legacy_builds/` | 原 CUDA 二进制与构建缓存，不纳入 Git |
| `../protocol/` | 数据身份、源版本、旧路径映射、重放工具和跨仓库锁 |

同盘目录搬迁保留 device/inode，并在旧路径保留兼容链接。旧源码树继续作为
对照来源保留；日常开发以新 fork 为准。原 Git 历史 bundle、工作树补丁和
包含疑似历史凭据的原始 README 放在本地 `results/migration/.../private/`，
不推送到公开远端。公开副本的两处凭据备忘清理有单独 hash 记录。

## 版本与修改时间

代码开发分支为两个 fork 的 `paper1/main`。文档、规约与轻量结果使用独立
本地 Git 仓库，分别推送到 Nerfstudio fork 的 `paper1/docs`、
`paper1/protocol`、`paper1/results` 分支；分支之间不混合目录树。

导入按原始修改日期分批提交，原始 mtime 保存在清单中，Git 作者/提交时间
记录实际迁入时间。完整文档顺序见 `../00_index/chronology.md`；包含新旧
路径、原时间、大小和 SHA256 的清单是 `../00_index/legacy_path_map.json`。
后续修正和新增兼容工具作为实际修改提交，不改写已发布历史来制造旧时间。

旧仓库的 1,796 个已跟踪文件、577 个未跟踪文件均有新的已跟踪位置。
检查工具是 `../protocol/tools/check_import_coverage.py`，验收结果是
`../protocol/migration/git_import_coverage.json`。原被忽略但必要的文档、PDF、
图表和证据包也已导入；文件不再因位于代码仓库外而失去版本管理。

## 功能检查

| 检查 | 结果 |
| --- | --- |
| 704 个算法/运行源码文件 | SHA256 与迁移前一致；README 脱敏单独记录 |
| 2,798 个文档文件 | SHA256 与迁移前一致 |
| 1,968 次目录/文件搬迁 | inode 与兼容链接检查通过 |
| 最终证据包声明的 418 个文件 | 全部存在，SHA256 一致 |
| 组合图的 36 张原图 | 原路径仍可访问，另有版本化输入副本 |
| 92 个 Hydra 实验 alias | 两边均解析成功，完整配置签名一致 |
| 四数据集 | 划分、frame IDs、图像、位姿与间距签名一致 |
| 两边 GPU/reference/结果测试 | 各 62 项通过，包含其他自实现渲染器的 22 项对照 |
| 两边 CPU PDF 检查 | 各 1 项通过 |
| 四数据集同一检查点重渲染 | 16 项预测/深度比较，最大绝对差值为 0 |
| 六项指标及迁出预训练权重 | 禁止下载时两边均可运行，数值完全一致 |
| 33,399 个数值/配置/报告快照 | 与原运行文件 SHA256 一致；一处公开 README 脱敏另行记录 |
| 冻结配置 CLI 重放 | 作业 88714 通过，PSNR/L1/SSIM 与原检查点评测一致，历史权重读取与新写入目录分离 |

主要 GPU 作业：88703（配置与四数据集训练回放）、88710（通过的 reference
测试）、88712（固定权重渲染及旧目录重跑对照）、88713（离线指标权重）。
其他自实现渲染器的补充对照作业是 88715，两边各 22 项通过。
初始测试遇到 Conda 激活与模块初始化差异、pytest 包名冲突和父目录插件
参数、计算节点缺少 PDF 工具；这些执行环境问题已处理，原算法未改写。

严格比较两次独立训练的逐索引权重未通过。CUDA 归约与动态密度控制路径
不保证逐位确定，旧目录自身重复训练也存在数值波动；本阶段没有用宽松
阈值把这项检查改为通过，也没有用一组对照宣称统计等价。
原严格失败记录与后续旧目录重复对照同时保留。

验收仅覆盖配置、文件、reference 测试和 12-step 训练/固定权重渲染；没有
重跑整套历史 30k 矩阵。238 条最终来源记录中，114 条仍引用存在的检查点，
其余仅有数值和已导出图像；文件存在的权重也不统一标为“已验证可续训”。
迁移不会恢复历史已删除权重，也不把新 fork SHA 写成原实验执行 SHA。

## 运行与重放

在原 `gaussian_splatting` 环境中，从 Nerfstudio fork 启动兼容入口：

```bash
cd /home/boywang25/paper1/nerfstudio
python -m nerfstudio.scripts.ultrasound_legacy experiment=<原alias>
```

默认数据、结果、检查点和 Torch 权重缓存均指向独立目录；显式路径 override
优先。克隆后可运行 `python ../protocol/tools/link_runtime.py` 恢复未跟踪链接。
数据、权重、原 ABI 二进制需另行提供，Git clone 不会下载这些大型资产。

复查历史实验时优先使用完整配置快照，避免当前 alias 默认值漂移：

```bash
python -m nerfstudio.scripts.ultrasound_replay <resolved_config.yaml> \
  --output <新的空run目录> --task dry_run

python -m nerfstudio.scripts.ultrasound_replay <resolved_config.yaml> \
  --output <新的空run目录> --task test \
  --checkpoint-dir <历史权重目录> --resume <step>
```

重放入口记录原配置/manifest 的 hash 和原执行来源；当前执行使用新的 run ID。
历史权重仅用于读取，新权重写到新的外置目录。测试可只加载保留的模型与
必要位姿状态，续训则要求完整 checkpoint 分片。旧恢复协议没有完整 RNG、
采样游标与密度控制累积状态，因此不宣称逐位续训等价。

可机器核对的结果见 `../protocol/migration/validation_summary.json`、
`../results/releases/migration_2026-10-17/`。完整 Slurm 日志和中间对照在
`../results/migration/2026-10-17/`。
