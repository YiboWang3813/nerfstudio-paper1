# OmniUltra Pipeline Skills

本文档定义项目从文档、代码、配置、脚本到实验输出和验证证据的流水线。它不规定通用开发、git/worktree 执行模式、文件范围或 Slurm 操作细节；那些分别见 `development_skills.md`、`git_workflow_skills.md`、`scope_skills.md` 和 `slurm_skills.md`。

## 1. 流水线总览

长期资产按职责分层：

```text
docs/           问题、假设、探索过程、结论、promotion 决策和生产结果说明
configs_hydra/  唯一可运行配置系统，提供 preproduction 与 production alias
methods/        Python 方法实现
ops/            CUDA/renderer/native extension
scripts/        数据处理、实验启动、结果汇总和维护脚本
outputs/        被 git ignore 的运行产物；生产结果可保留轻量证据
tests/          配置、方法、renderer、数学 reference 和回归测试
```

标准闭环：

```text
docs idea/report
  -> methods/ops code
  -> configs_hydra experiment alias
  -> scripts launch/summarize
  -> outputs run evidence
  -> docs conclusion + promoted config/script/code
  -> git commit
```

## 2. Docs 先行

中等以上实验应先有语义位置：

```text
docs/datasets/<dataset>/
docs/methods/baselines/<method>/
docs/methods/ours/<idea_name>/
docs/experiments/
docs/operations/
```

实验前文档不必很长，但至少能回答：

```text
why: 为什么做这个实验
what: 用哪个数据、方法、配置、metric
where: 对应 config alias、script、output root 在哪里
decision: 结果会如何影响代码、配置或论文结论
```

完成后必须把经验沉淀回文档。不要依赖保留整个 `outputs/` run 目录来保存记忆。

## 3. 配置生命周期

`configs_hydra/` 是唯一运行入口。新实验使用短 alias：

```bash
python run.py experiment=preproduction/ideas/ours/pose_uncertainty_prior/pose_prior
python run.py experiment=preproduction/baseline_reproduction/ultragray/main_10k_fast
python run.py experiment=production/comparisons/us_synthetic_recon_benchmark_v1/ours
```

生命周期目录：

```text
configs_hydra/experiment/preproduction/baseline_reproduction/<method>/<name>.yaml
configs_hydra/experiment/preproduction/ideas/ours/<idea>/<name>.yaml
configs_hydra/experiment/preproduction/dataset_preflight/<dataset>/<name>.yaml
configs_hydra/experiment/production/comparisons/<benchmark>/<method>.yaml
configs_hydra/experiment/production/ablations/<method_or_idea>/<name>.yaml
```

规则：

1. preproduction alias 用于 baseline 复现、idea 探索、dataset preflight、调参和诊断。
2. production alias 用于正式对比、正式消融、长期复查结果。
3. production alias 应尽量是已验证 preproduction alias 的 thin wrapper。
4. 临时参数优先用 command-line override，不为每个短跑新增 YAML。
5. 禁止恢复 `config_files/`、`python run.py --config ...` 或 `CONFIG=config_files/...`。

## 4. 输出目录和保留策略

推荐输出根：

```text
outputs/preproduction/baseline_reproduction/<method>/
outputs/preproduction/ideas/ours/<idea_name>/
outputs/preproduction/dataset_preflight/<dataset>/
outputs/preproduction/regression/<short_name>/
outputs/production/comparisons/<comparison_name>/
outputs/production/ablations/<method_or_idea>/<ablation_name>/
outputs/production/benchmarks/<benchmark_name>/
```

Hydra 自动写入 run 证据：

```text
.hydra/
.omniultra/
config_dump.yaml
resolved_config.yaml
overrides.yaml
run_manifest.yaml
checkpoints/
logs/
vis/
```

预实验产物默认可清理，尤其是大 checkpoint、重复失败 run、临时 profile 和临时可视化。清理前必须保留轻量证据：

```text
experiment alias:
key overrides:
output root:
representative run dirs:
metrics or qualitative evidence:
kept artifacts:
deleted artifacts:
decision:
promoted code/config/script:
```

生产实验必须保留：

```text
outputs/production/<kind>/<name>/README.md
outputs/production/<kind>/<name>/MANIFEST.yaml
outputs/production/<kind>/<name>/tables/*.csv|*.tsv|*.md
outputs/production/<kind>/<name>/notes/
```

大权重默认不进 git。需要长期追踪权重时，报告必须说明 checkpoint policy、选择 metric、路径或 artifact 位置、eval-only 命令。

### Checkpoint 外置存储

当 home 或项目同级 `outputs/` 存储紧张时，保持实验 run 目录仍在项目
`outputs/` 下，只把大 checkpoint 文件写到非 home 目录，并在 run 目录内
保留 `checkpoints` symlink。这样日志、配置快照、summary、vis 和脚本默认
仍按 `<run_dir>/...` 工作，读取 checkpoint 的旧路径
`<run_dir>/checkpoints/iteration_*.pth` 也不会断。

运行新实验时优先用 Hydra override：

```bash
python run.py \
  experiment=<alias> \
  system.checkpoint_root=/tmp/$USER/omniultra_checkpoints
```

也可以对单次 run 指定完整 checkpoint 目录：

```bash
python run.py \
  experiment=<alias> \
  system.checkpoint_dir=/tmp/$USER/omniultra_checkpoints/<run_name>/checkpoints
```

相关字段：

```text
system.output_dir        本地 run 目录，仍在 outputs/ 下
system.checkpoint_root   外置 checkpoint 根目录；按 output_root 相对路径自动派生 run checkpoint 目录
system.checkpoint_dir    真实 checkpoint 目录；显式设置时优先级最高
system.link_checkpoint_dir=true  在 output_dir/checkpoints 创建 symlink
```

`run_manifest.yaml` 和 `resolved_config.yaml` 会记录真实 `checkpoint_dir`。
检查一次 run 是否已外置：

```bash
ls -ld <run_dir>/checkpoints
readlink -f <run_dir>/checkpoints
```

迁移已有 run 时先 dry-run，再 apply：

```bash
python scripts/maintenance/externalize_checkpoints.py \
  --outputs-root outputs \
  --checkpoint-root /tmp/$USER/omniultra_checkpoints
python scripts/maintenance/externalize_checkpoints.py \
  --outputs-root outputs \
  --checkpoint-root /tmp/$USER/omniultra_checkpoints \
  --apply
```

注意：`/tmp` 在当前机器上是本地大盘，适合缓解 home quota；但如果集群会
清理 `/tmp`，长期生产 checkpoint 应放到用户有写权限且不会自动清理的非
home 持久目录。

### 4.1 Canonical Run 结构

每个可保留 run，无论 preproduction 还是 production，整理后应尽量收敛到：

```text
<run_dir>/
  .hydra/
  .omniultra/
  config_dump.yaml
  resolved_config.yaml
  overrides.yaml
  run_manifest.yaml

  logs/
    run_*.log
    scalars.tsv

  checkpoints/
    iteration_<best>_*.pth
    iteration_<last>_*.pth
    checkpoint_manifest.yaml

  eval/
    test_metrics.tsv              # if produced
    per_frame_metrics.tsv         # if produced
    selected_frames.txt           # if produced

  vis/
    item_<selected>/
    vis_manifest.yaml
```

单文件 checkpoint 的方法可以把保留文件命名为 `best.pth` 和
`last.pth`。如果一个方法拆成 `container`、`solver` 等多个 checkpoint
分片，保留原始 `iteration_<step>_<part>.pth` 命名，并在
`checkpoint_manifest.yaml` 中记录哪个 step 是 best、哪个 step 是 last。

### 4.2 Checkpoint Retention

训练过程中可以按 interval 暂存 checkpoint 用于断点恢复，但 run 完成后默认只长期保留：

1. best checkpoint：按主选择指标选择，默认是 `test/psnr` 最大的 step。
2. last checkpoint：训练结束或最大可用 checkpoint step。
3. `checkpoint_manifest.yaml`：记录 best/last step、选择 metric、保留文件和删除文件数量。

默认删除：

1. 中间 interval checkpoint，例如 `iteration_5000`、`iteration_10000` 等非 best/last step。
2. 重复失败 run 的 checkpoint。
3. preproduction 中已经写入报告且未 promotion 的 checkpoint。

只有以下情况才保留全量 interval checkpoint：

1. 正在分析训练 dynamics、collapse 或 checkpoint 选择策略。
2. 论文/报告明确需要中间 checkpoint 复查。
3. 用户或 manifest 明确标记 `checkpoint_policy=keep_all`。
4. 生产事故排查期间临时保留；排查后仍应重新清理。

### 4.3 Visualization Retention

scalar 和 per-frame metric 是轻量证据，应完整保留。全量测试图像默认不完整保留。

默认策略：

1. 保留 `logs/scalars.tsv` 的完整时间序列。
2. 保留 `eval/per_frame_metrics.tsv` 或其他 per-frame metric 文件，如果 run 产生了它们。
3. `vis/item_*` 默认只保留 10 个代表样本。
4. 若已有明确 paper figure selection，可保留对应 item，并在 `vis_manifest.yaml` 记录。

只有以下情况才保留 40 张或更多全量可视化：

1. 正在做 production figure selection。
2. 出现异常，需要完整视觉诊断。
3. 论文补充材料需要完整 qualitative grid。
4. 用户或 manifest 明确标记 `vis_policy=keep_all`。

### 4.4 Cleanup 工具

使用维护脚本整理已有 run：

```bash
python scripts/maintenance/cleanup_outputs.py --root outputs --dry-run
python scripts/maintenance/cleanup_outputs.py --root outputs --apply
```

脚本应先 dry-run，再 apply。默认行为是：

1. 保留 config、manifest、logs、scalars 和轻量 metric 文件。
2. checkpoint 只保留 best step 和 last step。
3. `vis/item_*` 只保留 10 个代表 item。
4. 写入 `checkpoints/checkpoint_manifest.yaml` 和 `vis/vis_manifest.yaml`。
5. 不清理声明为 `keep_all` 的 run。

## 5. 探索期 SOP

探索期目标是获得可用代码、配置和经验，不是保存所有中间实验。

探索期包括：

1. 复现 baseline 或外部方法。
2. 构建自己的新方法或新 idea。
3. 短跑、小规模 ablation、参数扫。
4. 调试 renderer、loss、metric、data loading、训练稳定性。

规则：

1. 用 preproduction alias 或已有 alias 加 override 启动。
2. 中间输出写到 `outputs/preproduction/...`。
3. 不长期保留失败 checkpoint。
4. 不把短跑小结果当成正式对比或正式消融。
5. 依赖 `resolved_config.yaml`、`overrides.yaml`、`run_manifest.yaml` 回溯细节。
6. 重要失败和负结果写回对应 `docs/methods/` 或 `docs/datasets/`。

探索报告至少记录：

1. 起点 alias。
2. 关键 override。
3. 代表性 run 目录。
4. 失败尝试和原因。
5. 有效设置和判断依据。
6. 最终是否 promotion。
7. 如果 promotion，生产代码、生产 alias、生产脚本在哪里。

## 6. Promotion Gate

探索内容进入生产期必须满足：

1. 代码整理到可维护状态。
2. 配置可用短 alias 表达，不依赖一串临时 override。
3. `task=dry_run` 能通过。
4. 相关 smoke、配置测试或 reference 测试能通过。
5. 报告写清楚为什么这个设置值得保留。
6. 生产 alias 名字表达实验意图。
7. 中间探索 checkpoint、临时 shell、临时参数扫表已删除、归档或明确不再依赖。

promotion 后保留：

1. 可维护代码。
2. 一个或少数几个 production alias。
3. 必要的生产脚本。
4. 必要测试。
5. 文档中的关键经验和结论。

## 7. 生产期 SOP

生产期目标是产出可复查、可汇总、可写论文/报告的结果。生产期不应该再做方法设计或随手调参。

前提：

1. 参与实验的方法都有 production-ready alias。
2. 数据集、split、seed、iteration、metric、eval protocol 明确。
3. Slurm 脚本只依赖 `EXPERIMENT` 和 `OVERRIDES`，不依赖旧 YAML 路径。
4. 汇总脚本明确读取哪个 run 目录和哪些 metric key。
5. 报告能复原 alias、override、输出目录和汇总表。

生产对比流程：

1. 写对比矩阵：dataset/sweep、method、alias、override、seed、iteration、resume。
2. 对每个 method 做 `task=dry_run`。
3. 使用 `scripts/production/comparisons/<comparison_name>/` 批量提交。
4. 为汇总稳定，可覆盖 `hydra.run.dir=${OUTPUT_ROOT}/${EXP_NAME}`。
5. 从 `logs/scalars.tsv` 或最终 eval 输出读取指标。
6. 汇总表写到 `outputs/production/comparisons/<comparison_name>/tables/`。
7. README/MANIFEST/notes 记录 run selection、known issues 和结论。

正式消融流程：

1. 主方法已有 production-ready alias。
2. 消融项来自探索期经验。
3. 每个 ablation 只改变明确因素。
4. 短期 ablation 优先用 override；长期复用才新增 production ablation alias。
5. 输出根使用 `outputs/production/ablations/<method_or_idea>/<ablation_name>/`。

## 8. Eval、Benchmark 和回归

Final eval 必须记录 checkpoint policy：

```text
last / best / fixed_iter / external
metric used for best
checkpoint path or artifact location
eval-only command
```

多 seed 和跨 sweep 生产结果必须保留 per-run 表和 aggregate 表，并记录失败 run 的状态和原因。

生产 benchmark 必须固定硬件、CUDA/module、环境、分辨率、iteration、warmup、重复次数和 profiler 开关。探索期 profile 数字不能直接当生产 benchmark。

代码或配置改动后的最小回归：

```bash
python -m py_compile run.py configs/setup.py configs/root.py
python run.py experiment=<alias> task=dry_run system.device=cpu engine.iterations=1
bash -n scripts/<owner>/<script>.sh
```

修改共享配置加载或 schema 时，运行相关配置测试。修改 renderer、CUDA op 或核心数学时，运行对应 reference/shape/smoke 测试。

## 9. 常见反模式

禁止：

1. 在根目录恢复 `config_files/`。
2. 新脚本里写 `python run.py --config ...`。
3. 新脚本里使用 `CONFIG=...` 表示实验配置。
4. 把 experiment 文件名写成长参数串。
5. 手动拼 `checkpoints/<method>/<exp_tag>` 作为默认生产输出。
6. 把一次性调参覆盖项固化成大量近似重复 YAML。
7. 跑完实验不记录 run 目录、override 和关键指标。
8. 把探索期中间 checkpoint 和临时启动脚本当成生产资产长期保留。
