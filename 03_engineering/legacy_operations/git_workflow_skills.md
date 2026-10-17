# OmniUltra Git Workflow Skills

本文档定义不同业务场景如何使用 git branch、worktree 和 main。它只管执行模式，不管具体文件范围；文件范围见 `scope_skills.md`，开发规范见 `development_skills.md`。

## 1. 总原则

1. `main` 表示当前项目主线，production 实验默认只从 `main` 启动。
2. 探索性 ours idea 用普通 branch 隔离。
3. baseline 复现用同级 worktree 隔离。
4. production 比较和 production ours 消融在配置冻结后从 `main` 跑。
5. 任何 production 中发现的代码 bug，都先开修复分支或对应 worktree 修复，合回 `main` 后再重跑。
6. 不用 git 追踪大输出、checkpoint、wandb、logs 和 build 产物。

## 2. Ours Idea Exploration

用途：

```text
新的 ours 创新点尝试、idea 可行性验证、预实验调参、探索性代码路径。
```

推荐模式：

```bash
git switch main
git pull --ff-only  # if remote is used and network is available
git switch -c ours-idea-<short_name>
```

当前仓库已有大量同类分支，例如：

```text
ours-idea-frequency-edge
ours-idea-edge-allocation
ours-idea-reliability-anchors
ours-idea-slice-thickness
ours-idea-structure-speckle
dev/ours-*
```

规则：

1. 一个 idea 一个 branch，分支名表达语义，不用日期当主语义。
2. 分支里允许修改对应 idea 的 docs、configs、scripts、methods、ops 和 tests。
3. 预实验输出落到 `outputs/preproduction/ideas/ours/<idea>/`，输出本身不进 git。
4. 探索结束后必须把结论写回 `docs/methods/ours/<idea>/`。
5. 只有验证清楚的代码、配置、脚本和文档才整理回 `main`。
6. 不把失败 checkpoint 或大中间产物作为保留证据；保留轻量指标、manifest 和结论。

适合继续用 branch，而不是 worktree 的原因：

```text
ours idea 通常共享当前主线的 ours 实现，需要频繁对比和合并；普通 branch 足够隔离，成本低。
```

只有在一个 idea 需要长期并行训练、重建 CUDA 或和 main 同时开发时，才临时增加 worktree。

## 3. Baseline Reproduction

用途：

```text
复现 UltraGauss、UltraGray、UltraNeRF、GaussianPile 等 baseline。
```

推荐模式：

```bash
git worktree add ../dev-<baseline>-renderer -b dev-<baseline>-renderer main
```

当前仓库同级目录已有：

```text
../dev-ultragauss-renderer
../dev-ultragray-renderer
../omniultra-dev-ours-renderer
```

规则：

1. baseline 复现默认使用同级 worktree。
2. worktree 名和 branch 名应包含 baseline 名。
3. baseline 可以在 worktree 内独立处理 renderer、第三方接口、配置适配和诊断脚本。
4. 复现完成后，只把主线需要保留的代码、配置、脚本、测试和文档合回 `main`。
5. worktree 中的临时输出、编译产物和失败 checkpoint 不合回。
6. 如果复现需要修改共享入口，例如 `run.py`、`configs/base/models.py`，必须保持最小注册改动。

适合用 worktree 的原因：

```text
baseline 复现经常碰 renderer、ops、third-party 适配和长时间实验；worktree 可以让多个 baseline 并行，不污染主线工作目录。
```

## 4. Production Comparison

用途：

```text
同一个数据集、同一个协议下，多方法生产级正式对比。
```

推荐模式：

```text
在 main 上启动 production run。
```

规则：

1. production comparison 只使用已经确认稳定的方法代码和 production alias。
2. production alias 放在 `configs_hydra/experiment/production/comparisons/<benchmark>/`。
3. production 脚本放在 `scripts/production/comparisons/<benchmark>/`。
4. 输出落到 `outputs/production/comparisons/<benchmark>/`。
5. 不在 production run 中夹带探索性代码修改。
6. 如果发现 bug，停止把该结果当正式结果，开修复分支或相关 worktree，修完合回 `main` 后重新跑。

## 5. Production Ours Ablation

用途：

```text
ours 已验证创新点在生产级配置下的正式消融。
```

推荐模式：

```text
在 main 上启动 production ablation run。
```

规则：

1. 消融基于已经 promotion 的 ours idea。
2. production ablation alias 放在 `configs_hydra/experiment/production/ablations/<idea_or_method>/`。
3. production ablation 脚本放在 `scripts/production/ablations/<idea_or_method>/<ablation_name>/`。
4. 输出落到 `outputs/production/ablations/<idea_or_method>/<ablation_name>/`。
5. 只改变消融矩阵需要改变的 production 配置，不重新引入探索性代码。
6. 如果消融暴露方法 bug，按 bugfix 流程处理，再重跑正式消融。

## 6. Dataset Cleaning Or Preflight

用途：

```text
数据清洗、split 生成、metadata 检查、dataset preflight。
```

推荐模式：

```text
小型脚本和文档修改可在 main 上做；会影响训练协议或数据读取语义时开 branch。
```

规则：

1. 清洗脚本放在 `scripts/datasets/<dataset>/`。
2. 数据集文档放在 `docs/datasets/<dataset>/`。
3. preflight alias 放在 `configs_hydra/experiment/preproduction/dataset_preflight/<dataset>/`。
4. 不把清洗产生的大数据副本提交 git。
5. 如果改变 dataset split 或 evaluation protocol，必须在文档中写清楚影响范围。

## 7. Shared Bugfix Or Integration

用途：

```text
修复 engine、scene、gaussian、utils、shared config、公共 registry、公共 CUDA helper。
```

推荐模式：

```bash
git switch main
git switch -c fix/<short_name>
```

规则：

1. 先确认不是某个 method 可以局部解决的问题。
2. 改动保持最小，附带能覆盖受影响路径的 smoke 或 regression 验证。
3. 合回 `main` 后，受影响的 production run 需要重新生成结果。
4. 共享改动不应混在 production run 提交里。

## 8. 提交边界

每次提交应表达一个清楚业务目的：

```text
Explore <ours idea>
Reproduce <baseline>
Promote <idea> production config
Add <benchmark> production comparison scripts
Fix <shared component> <bug>
Document <dataset/protocol/result>
```

提交前必须确认：

```text
git status --short
git diff --name-only
```

如果同一工作目录里同时存在无关改动，提交时只 stage 当前业务相关文件。
