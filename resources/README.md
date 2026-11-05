# 资料与来源

数据正文在 shared/datasets，本目录只保存说明。PDF 统一放 [papers](papers)，研究笔记包统一放 [research-notes](research-notes)。

- [Dataset Descriptions for Production Experiments](dataset-descriptions.md)
- [spine_phantom Dataset Notes](dataset-spine-phantom-overview.md)
- [synthetic_liver Dataset Notes](dataset-synthetic-liver-overview.md)
- [tdsc_abus_2023 Dataset Notes](dataset-tdsc-abus-2023-overview.md)
- [tus_rec_2024 Dataset Notes](dataset-tus-rec-2024-overview.md)

## 文献

- [3dgs-fruitninja-paper](papers/3dgs-fruitninja-paper.pdf)
- [coarse-pose-paper](papers/coarse-pose-paper.pdf)
- [dclnet-paper](papers/dclnet-paper.pdf)
- [fima-paper](papers/fima-paper.pdf)
- [fracture-gs-paper](papers/fracture-gs-paper.pdf)
- [free-splatter-paper](papers/free-splatter-paper.pdf)
- [gaussianpile-paper](papers/gaussianpile-paper.pdf)
- [grrt-paper](papers/grrt-paper.pdf)
- [hogs-paper](papers/hogs-paper.pdf)
- [litags-paper](papers/litags-paper.pdf)
- [long-splat-paper](papers/long-splat-paper.pdf)
- [loop-sparse-gs-paper](papers/loop-sparse-gs-paper.pdf)
- [mags-paper](papers/mags-paper.pdf)
- [nerfus-dagli24a](papers/nerfus-dagli24a.pdf)
- [no-pose-no-problem-paper](papers/no-pose-no-problem-paper.pdf)
- [nopo-room-paper](papers/nopo-room-paper.pdf)
- [pcr-gs-paper](papers/pcr-gs-paper.pdf)
- [phys-splat-paper](papers/phys-splat-paper.pdf)
- [robust-pose-estimation-paper](papers/robust-pose-estimation-paper.pdf)
- [sdhnet-paper](papers/sdhnet-paper.pdf)
- [steam-gs-paper](papers/steam-gs-paper.pdf)
- [topo4d-paper](papers/topo4d-paper.pdf)
- [tus-rec-2024-summary](papers/tus-rec-2024-summary.pdf)
- [ultragray-paper](papers/ultragray-paper.pdf)
- [ultrags-paper](papers/ultrags-paper.pdf)
- [ultranerf-paper](papers/ultranerf-paper.pdf)

## 当前资产来源

- [冻结输入包与 SHA256](provenance/evidence-packages.json)
- [原 results Git 历史](provenance/paper-evidence.json)
- [当前目录迁移映射](provenance/artifact-storage-map.json)

完整逐文件源清单在 outputs/provenance/source-catalog，保留原始记录身份；不将大规模 JSONL 当作文档日常文件跟踪。

文献 PDF 与研究笔记 ZIP 仅本机保留，不再进入 Git；题录文件名、大小与 SHA256 见 [literature-index.json](literature-index.json)。

详细迁移日志、清单和回执已分类到本目录 maintenance-assets（忽略）与 provenance（轻量记录）；原 migration 顶层目录已删除。旧 Git 证据数据库已由逐项验证的论文 run ZIP 取代。
