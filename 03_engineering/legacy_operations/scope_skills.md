# OmniUltra Scope Skills

本文档定义不同业务场景默认允许修改哪些文件。它不决定使用 branch 还是 worktree；git 执行模式见 `git_workflow_skills.md`。通用开发和越界升级见 `development_skills.md`。

## 1. 总原则

1. 先确定业务类型，再确定可修改路径。
2. 默认只修改 owning 目录。
3. 修改公共框架、其他方法、其他 dataset 或 production 共享协议前必须升级说明。
4. `outputs/**` 默认不进 git，但 run 目录结构和保留策略必须符合 `pipeline_skills.md`。
5. `logs/**`、`wandb/**`、checkpoint、build 产物和缓存不进 git。

## 2. Ours Idea Exploration

适用：

```text
ours 新 idea、预实验、消融前探索、可行性验证。
```

默认允许修改：

```text
docs/methods/ours/<idea>/**
configs_hydra/experiment/preproduction/ideas/ours/<idea>/**
scripts/methods/ours/<idea>/**
tests/methods/ours/**
tests/ops/ours_renderer/**
```

按需允许修改：

```text
methods/ours/**
ops/ours_renderer/**
configs_hydra/model/ours*.yaml
```

输出位置：

```text
outputs/preproduction/ideas/ours/<idea>/**
```

默认不应修改：

```text
methods/<baseline>/**
ops/<baseline_renderer>/**
configs_hydra/experiment/production/**
scripts/production/**
engine/**
scene/**
gaussian/**
utils/**
ops/common/**
```

promotion 后才允许把稳定 alias 或脚本整理到 production 目录。

## 3. Baseline Reproduction

适用：

```text
baseline 复现、baseline renderer 适配、baseline 诊断和调参。
```

默认允许修改：

```text
docs/methods/baselines/<baseline>/**
configs_hydra/experiment/preproduction/baseline_reproduction/<baseline>/**
scripts/methods/baselines/<baseline>/**
tests/methods/<baseline>/**
tests/ops/<baseline_renderer>/**
```

按需允许修改：

```text
methods/<baseline>/**
ops/<baseline_renderer>/**
configs_hydra/model/<baseline>.yaml
```

输出位置：

```text
outputs/preproduction/baseline_reproduction/<baseline>/**
```

最小 integration 可修改：

```text
run.py
configs/base/models.py
```

条件：

```text
只做 baseline 注册、导入或 config registry 接入，不改变共享执行语义。
```

默认不应修改：

```text
methods/ours/**
ops/ours_renderer/**
configs_hydra/experiment/production/**
scripts/production/**
engine/**
scene/**
gaussian/**
utils/**
ops/common/**
```

## 4. Dataset Cleaning Or Preflight

适用：

```text
数据清洗、split 生成、metadata 检查、dataset preflight、dataset 文档。
```

默认允许修改：

```text
docs/datasets/<dataset>/**
scripts/datasets/<dataset>/**
configs_hydra/experiment/preproduction/dataset_preflight/<dataset>/**
```

按需允许修改：

```text
configs_hydra/scene/<dataset>.yaml
configs_hydra/data/<dataset>.yaml
tests/datasets/<dataset>/**
```

输出位置：

```text
outputs/preproduction/dataset_preflight/<dataset>/**
```

默认不应修改：

```text
methods/**
ops/**
configs_hydra/experiment/production/**
scripts/methods/**
scripts/production/**
```

如果 dataset 改动影响 evaluation protocol、split 或所有方法共享的数据读取语义，必须在文档中说明影响，并按 integration 处理。

## 5. Production Comparison

适用：

```text
同一 benchmark 下多方法生产级正式对比。
```

默认允许修改：

```text
docs/experiments/**
docs/methods/baselines/<baseline>/**        # only final result notes when relevant
docs/methods/ours/<idea>/**                 # only final result notes when relevant
configs_hydra/experiment/production/comparisons/<benchmark>/**
scripts/production/comparisons/<benchmark>/**
```

输出位置：

```text
outputs/production/comparisons/<benchmark>/**
```

默认不应修改：

```text
methods/**
ops/**
configs_hydra/experiment/preproduction/**
scripts/methods/**
engine/**
scene/**
gaussian/**
utils/**
```

如果 production comparison 暴露代码 bug，先走 `git_workflow_skills.md` 的 shared bugfix 或 method fix 流程，修完合回 `main` 后重跑。

## 6. Production Ours Ablation

适用：

```text
ours 已验证创新点的生产级正式消融。
```

默认允许修改：

```text
docs/methods/ours/<idea>/**
docs/experiments/**
configs_hydra/experiment/production/ablations/<idea_or_method>/**
scripts/production/ablations/<idea_or_method>/<ablation_name>/**
```

输出位置：

```text
outputs/production/ablations/<idea_or_method>/<ablation_name>/**
```

默认不应修改：

```text
methods/**
ops/**
configs_hydra/experiment/preproduction/**
scripts/methods/**
engine/**
scene/**
gaussian/**
utils/**
```

只有当 ablation 需要把已验证 idea 的开关、参数或默认值 promotion 到正式模型配置时，才可修改：

```text
configs_hydra/model/ours*.yaml
methods/ours/configs.py
```

这类改动应单独提交，不和 run 结果整理混在一起。

## 7. Shared Integration

适用：

```text
公共入口、注册、engine、scene、gaussian、utils、shared CUDA helper、跨方法协议。
```

默认需要显式说明后才修改：

```text
run.py
configs/base/**
configs/root.py
configs/setup.py
configs_hydra/engine/**
engine/**
scene/**
gaussian/**
utils/**
ops/common/**
```

要求：

1. 说明为什么不能在 owning method 内局部解决。
2. 列出受影响方法和生产实验。
3. 加最小 smoke、config 或 regression 验证。
4. 不和 idea 探索、baseline 复现或 production run 整理混成一个提交。

## 8. Review 和只读调研

Review 默认不改文件。优先检查：

```text
越界改动
公共框架风险
配置字段不一致
Hydra alias 路径不一致
注册名不一致
缺测试
CUDA/性能风险
实验复现风险
outputs 保留策略不符合 pipeline
```

只读调研默认不修改文件、不安装依赖、不重建 extension、不提交 git。输出时列出关键路径、类、函数和建议。
