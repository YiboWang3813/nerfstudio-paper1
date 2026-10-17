# OmniUltra Development Skills

本文档定义 agent 在本项目中如何正确开发。它只约束通用开发行为、修改边界判断、验证和交付；业务对应的 git/worktree 模式见 `git_workflow_skills.md`，不同业务允许修改的文件范围见 `scope_skills.md`，实验流水线见 `pipeline_skills.md`，Slurm 执行见 `slurm_skills.md`。

## 1. 规则地图

按任务类型阅读：

```text
docs/operations/development_skills.md   通用开发规范、越界升级、交付检查
docs/operations/git_workflow_skills.md  idea、baseline、production 的 git/worktree 规范
docs/operations/scope_skills.md         不同业务类型允许修改的文件范围
docs/operations/pipeline_skills.md      docs/configs/scripts/outputs 的实验流水线
docs/operations/slurm_skills.md         GPU、Slurm、CUDA 编译、训练监控
```

阅读选择：

1. 只读调研：读本文件和相关源码/文档。
2. 修改代码、配置、脚本或文档：读本文件、`git_workflow_skills.md`、`scope_skills.md`。
3. 新增或调整实验 alias、输出、promotion 记录：读 `pipeline_skills.md`。
4. 提交训练、评估、CUDA 编译、GPU 测试：读 `slurm_skills.md`。

## 2. 开发入口

开始前先确定任务属于哪一类：

```text
ours idea exploration
baseline reproduction
dataset cleaning or preflight
production comparison
production ours ablation
integration or shared framework change
read-only research
review
```

确认任务类型后：

1. 用 `git_workflow_skills.md` 确认当前分支、worktree 或 main 是否合适。
2. 用 `scope_skills.md` 确认可修改路径。
3. 用 `pipeline_skills.md` 确认 docs、configs、scripts、outputs 的闭环位置。
4. 只读调研和 review 默认不修改文件。

## 3. 基本开发规则

项目是 Hydra-only。新实验和新脚本必须使用：

```bash
python run.py experiment=<lifecycle>/<category>/<name>
```

禁止恢复旧入口：

```bash
python run.py --config ...
CONFIG=config_files/...
```

新增或修改功能时：

1. 先读已有文档和相邻实现，不凭空新增平行结构。
2. 优先把代码放在 owning method/ops 目录。
3. 配置默认值放在 method dataclass 和 `configs_hydra/model/<method>.yaml`。
4. 实验 alias 放在 lifecycle-aware 的 `configs_hydra/experiment/**`。
5. 启动脚本放在 `scripts/methods/**` 或 `scripts/production/**` 的 owning 目录。
6. 结果和运行证据落到 `outputs/preproduction/**` 或 `outputs/production/**`。
7. 经验、结论和 promotion 决策必须回写到 `docs/**`。

## 4. 越界升级

如果任务需要修改当前业务不拥有的公共层或其他方法包，先暂停并说明：

```text
需要修改的越界文件：
- <path>

原因：
- <why this is required>

最小改动方案：
- <minimal change>

可能影响：
- <affected methods or runtime behavior>
```

常见需要升级为 integration 的情况：

1. 新增方法注册到 `configs/base/models.py`。
2. 新增 state/callback 并需要 `run.py` 导入注册。
3. 修改 `engine` 调度语义。
4. 修改 `scene` 数据结构、sampler 或 metadata 解析。
5. 修改 `gaussian` 通用容器、solver 或 scheduler。
6. 修改 `ops/common` 或跨 renderer CUDA helper。
7. 修改 shared loss/metric 语义。
8. 修改 production alias 的公共协议字段。

## 5. 方法和 Ops 规则

方法主体代码写在：

```text
methods/{method_name}/**
```

method-owned renderer 或自定义算子写在：

```text
ops/{method_name}_renderer/**
```

现有对应关系：

```text
methods/ours       -> ops/ours_renderer/**
methods/ultragauss -> ops/ultragauss_renderer/**
methods/ultragray  -> ops/ultragray_renderer/**
methods/ultrags    -> ops/ultrags_renderer/**
methods/s2s        -> ops/s2s_shader/**
```

实现要求：

1. 复用 method 包内已有风格。
2. 临时代码放到对应 `scripts/`、preproduction config 或探索文档，不混进核心路径。
3. 修改 renderer API 时同步更新 method 调用点和测试。
4. 修改配置字段时同步更新 dataclass、Hydra model yaml、实验 alias 和配置测试。
5. CUDA code 改动后如果结果没变，优先怀疑 extension 没重建。

## 6. 验证要求

最小验收：

```text
只改文档：git diff --check；检查路径和规则是否与仓库一致。
只改 method Python：相关 import/config build 或 tests/methods。
新增或修改 config：字段存在于 dataclass；dry_run 通过。
修改 CUDA/Ops：重建对应 ops；跑 reference/shape/smoke 测试之一。
新增 Slurm runner：bash -n；确认 Hydra alias 和输出根。
Integration 接入：注册名一致；smoke alias 可 dry_run。
```

最小配置检查：

```bash
python run.py experiment=<lifecycle>/<category>/<name> task=dry_run system.device=cpu engine.iterations=1
```

训练、评估、CUDA 编译、GPU benchmark 通过 Slurm 跑在计算节点，具体见 `slurm_skills.md`。

## 7. Git 和交付

提交规则：

1. 只提交和任务相关的文件。
2. 不提交 checkpoint、日志、缓存、wandb、build 产物、`.egg-info`。
3. 如果脚本需要 `chmod +x`，mode change 和内容一起提交。
4. 提交前检查 `git diff --name-only`、`git status --short`。

常见不应进入 git 的文件：

```text
__pycache__/**
*.pyc
.pytest_cache/**
wandb/**
logs/**
outputs/**
checkpoints/**
build/**
*.egg-info/**
```

交付前自检：

```text
[ ] 已确认业务类型和 git/worktree 模式
[ ] 已确认修改范围符合 scope 规则
[ ] 没有无授权修改公共框架层或其他方法
[ ] 已运行最小必要验证或说明无法运行原因
[ ] 已说明剩余风险
[ ] 通过验证的代码、配置、脚本和文档修改已提交 git
```
