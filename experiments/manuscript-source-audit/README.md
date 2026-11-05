# Mac 文稿图表来源核对

金标准：workspace/exchange/inbox/manuscript-for-216-20261105-155440，Mac 原提交 `9d2649c1a6910fcf190f876ad3a283bce74d931b`；39 个传输文件 SHA256 全部通过。未修改收到的批次。

| 文稿对象 | 核对结果 |
| --- | --- |
| methods_overview.pdf | 作者示意图，PDF 已保留；可编辑源未提供 |
| results_ablation_afgm.pdf | 本地原 PDF 与 Mac SHA256 完全一致 |
| results_ablation_cgas.pdf | 本地原 PDF 与 Mac SHA256 完全一致 |
| results_ablation_plgr.pdf | 本地原 PDF 与 Mac SHA256 完全一致 |
| results_comparison_all_datasets.pdf | 本地原 PDF 与 Mac SHA256 完全一致 |
| comparison_four_datasets.tex | 224 个显示数值与冻结原始数据一致 |
| ablation_joint_components.tex | 48 个显示数值与冻结原始数据一致 |
| ablation_afgm.tex | 24 个显示数值与冻结原始数据一致 |
| ablation_plgr.tex | 30 个显示数值与冻结原始数据一致 |
| ablation_cgas.tex | 30 个显示数值与冻结原始数据一致 |

总计 5 图、5 表、356 个显示数值。主表旧 exchange TeX 的 Ours 行已过时，已撤出当前工程；AFGM 的旧多 sweep 扩展面板和 H4 群体曲线未出现在 Mac 稿中，不再保留其独立输出。

来源链：Mac 图表 → 冻结 source TSV/配置 → 原 run 的最终逐帧指标、配置、数量与耗时记录。保留 774 个相关 run 的 5155 个原始文件，ZIP 内逐项 SHA256 与原件一致；保留 36 个实际定性面板。输入包含当前生成器实际读取的 270 个源文件（含共享审计依赖），不再整包保存旧图形变体。

已收拢并清理 155138 个旧散落文件（其中论文所需原件已验证打包），约 14.48 GiB；不触碰 shared/datasets、其他项目或交换批次。

本机按对应 endo 风格和修正后的 Ours 来源生成器重建，数值来源一致；由于字体/排版环境不同，生成 PDF 并非与 Mac 像素完全相同。Mac 原始图 PDF 保留为正式版本，未被本机生成版覆盖。

- [逐项核对](audit.json)
- [保留与删除清单](retention.json)
- [生成版与原 PDF 的栅格差异](regeneration.json)
- [原始 run 索引](../../resources/provenance/paper-evidence.json)

最终 outputs 剩 137 个物理文件、169.24 MiB；详见 [最终验收](final-validation.json)。Git 历史已本地清理，旧新提交映射在 resources/provenance，远端没有强推。
