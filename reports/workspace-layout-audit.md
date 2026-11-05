# Paper1 工作区目录与 Git 负载审计

审计日期：2026-11-05。本轮仅检查和记录，不移动或删除资产、不改代码、不重写 Git 历史。全量目录统计不跟随软链接，因此未遍历 shared/datasets 或其他项目。

结论：顶层职责已基本建立，当前 Git 没有海量运行文件，也没有权重、数据正文或编译产物。但轻量文档要求尚未完全满足，仍有文献重文件、旧来源代码及过时规则需要收敛。

| 仓库 | 审计时当前跟踪文件数 | 当前跟踪内容总量 | 最大当前文件 |
| --- | ---: | ---: | --- |
| docs | 397 | 259.65 MiB | GaussianPile 文献 PDF，47.16 MiB |
| nerfstudio | 1,332 | 76.67 MiB | 上游可视化 notebook，14.65 MiB |
| gsplat | 773 | 30.48 MiB | 上游 training.gif，20.21 MiB |

上表为添加本审计报告之前的统计，不包含 .git；三个仓库合计 2,502 个当前跟踪文件，不是此前数万或十万文件的状态。当前没有超过 50 MiB 的单个跟踪文件，但存在多项超过 10 MiB 的内容，不能称为完全轻量。

## 已清楚的边界

- nerfstudio、gsplat 是两个源码仓库；docs 是研究文档和图表工程仓库。
- outputs 没有 Git 仓库，承担 runs、权重、日志、逐帧指标、渲染与批量导出。
- results 顶层目录和兼容链接均不存在。旧 results Git 数据库保留在 migration/git-history/results.git，不再作为日常工作树。
- workspace/exchange 是最新用户授权的唯一交换缓冲区，不能视为待清理输出。
- .venv 是项目入口覆盖层。数据正文、共享环境与缓存不迁入项目 Git。
- 三个仓库没有被跟踪的 .ckpt、.pt、.pth、.so、.o、.a 或 .npy；被跟踪软链接均可解析。gsplat 的约 2 MiB NPZ 是上游测试样例，不是实验数据集正文。

## 仍需整理的问题

1. **docs 仍跟踪重文献资产。** 26 个 PDF 与 4 个研究笔记 ZIP 被 Git 跟踪。最大 PDF 47.16 MiB，另外还有 25.66、18.93、14.90 MiB 等文献。建议保留本地文件，但从当前 Git 跟踪中退出；文档只跟踪题录、文件路径、来源和 SHA256，并使用精确忽略规则。
2. **代码仓库有较大文档或迁移资产。** Nerfstudio 的 notebook 含较重的可视化输出；numeric_snapshot_manifest.json 为 12.61 MiB，import_commits_results.json 为 5.70 MiB。gsplat 的上游 GIF 为 20.21 MiB。需要区分上游源文件与项目产物：可清空 notebook 的生成输出并保留完整原件；大迁移清单归 outputs/provenance；文档媒体按明确路径本地保留和忽略。
3. **旧代码参考仍占据代码仓库。** nerfstudio/paper1_legacy 下有 705 个当前跟踪文件，gsplat/ultrasound_ops 下有 92 个。原生运行代码没有导入前者，但测试和配置转换工具仍引用其少数路径；不能直接删除目录而不处理这些引用。另有约 6,000 份忽略的源端参考文件位于 nerfstudio/experiments 下，虽未被当前 Git 跟踪，物理位置仍容易与活动工程混淆。
4. **规则和迁移入口尚未完全统一。** 两份代码仓库 AGENTS.md 仍包含旧分支、失效的 docs/03_engineering 路径及旧 Slurm 规则，虽然标明新规优先，仍会增加误读。migration 根目录有 146 项，混合一次性脚本、详细日志和回执；宜保留简洁入口，其余按 scripts、logs、receipts 等明确职责整理，并同步引用。

## 小文件密集位置

- outputs/eehpc：148,130 个文件，约 11.86 GiB，均在当前三个 Git 工作树之外，是历史运行证据，不是随意可删缓存。
- outputs/eehpc-logs：4,598 个文件，约 629 MiB。
- nerfstudio/experiments：6,588 个物理文件，其中绝大部分为忽略的源端参考资料，不能将物理文件数误报为 Git 当前跟踪数。
- migration 中还有数个 100–170 MiB 的 JSONL 清单，没有进入上述三个当前 Git 仓库。

## Git 历史与当前跟踪不同

当前工作树退出跟踪，不等于旧提交里的大对象被清除。检查了三个活动仓库和保留的 results Git 数据库：历史仍有旧 PDF、notebook、GIF、清单和旧源树记录。当前 .git 大小约 docs 288.8 MiB、Nerfstudio 302.4 MiB、gsplat 36.8 MiB；保留的 results.git 约 382.3 MiB。

这些历史承担来源和固定证据 tag 的保全。本轮不做 filter-repo、强推或删除 Git 对象；若后续要求缩减整个仓库历史，需要单独设计来源、旧提交引用与远端协作的保留方式。不能将普通忽略操作描述成历史空间已经回收。

详细机器清单位于 paper1/migration/workspace-layout-audit.json，包含逐仓库大文件、目录分布和物理文件统计。
