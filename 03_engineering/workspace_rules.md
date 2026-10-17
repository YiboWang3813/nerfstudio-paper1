# Paper1 工作区规约

## 存储与版本边界

`nerfstudio/` 与 `gsplat/` 是两个专属 fork，开发分支均为 `paper1/main`。
`docs/`、`protocol/`、`results/` 使用独立本地 Git 仓库，远端分别保存在
Nerfstudio fork 的 `paper1/docs`、`paper1/protocol`、`paper1/results` 分支。
这些分支保存不同目录树，不合并到模型代码主分支。

数据、检查点、日志、运行输出和编译缓存的实际内容均位于代码仓库外。
代码中的未跟踪兼容链接仅供旧实验解析路径。旧项目路径也保留兼容链接，
便于复查原始 manifest；历史记录中的绝对路径不做批量字符串替换。

## 迁移与框架适配

`nerfstudio/paper1_legacy/` 保存原运行框架、配置、方法、脚本和测试。
`gsplat/ultrasound_ops/` 保存各自实现 CUDA 后端及共享头文件。
`gsplat.ultrasound` 提供显式后端加载接口；旧包名继续作为兼容接口。
此次搬迁不改变物理模型、优化规则、seed、split、指标或正式参数。

当前已验证路径优先使用原 Python 3.7 / PyTorch 1.12 环境。
原生 Nerfstudio Model/Pipeline 和现代 ABI 下的 CUDA 编译需独立验证，
不能将旧运行框架的成功测试当作原生 Nerfstudio 移植完成。

兼容启动命令，在原环境中运行：

```bash
cd /home/boywang25/paper1/nerfstudio
python -m nerfstudio.scripts.ultrasound_legacy experiment=<原alias>
```

启动器默认将新输出写到 `../results/runs/`、权重写到
`../checkpoints/runs/`，并从 `../data/` 加载数据。显式 Hydra override 优先。
已有生产脚本仍可从 legacy 目录运行；重新执行前必须选择新 run 目录，
不覆盖历史配置、日志和数值表。

历史实验优先使用 `python -m nerfstudio.scripts.ultrasound_replay` 读取完整
`resolved_config.yaml`，而非假设今天的 alias 与当时默认值相同。
该入口拒绝覆盖非空 run 目录，将历史权重读取与新权重写入分开；测试只
要求模型/位姿状态，恢复训练要求所有必要分片。

## 实验证据与续训

每个 run 保存数据/split、算法执行 SHA、有效配置、环境和 binary 身份。
图表必须能反查原始 run/frame、未舍入数据、绘图代码版本和裁剪/显示参数。
已有 evidence package 保留原始 hash，新的图形修订生成新 revision。

权重状态区分：仅指标/已导出图像、存在可用模型、完整可恢复训练。
文件存在本身不等于已验证可恢复；旧 manifest 不补写虚构的模型身份。
已删除权重不能由迁移恢复，新的视角渲染或续训可能需要重新训练。

运行 GPU 测试、重建算子或训练必须通过 Slurm；原环境和原 binary 的
保留副本不原地升级。新二进制使用独立 build cache 和环境身份。

## 文档与提交

保留方法的历史名称和目录映射；名称含 baseline 不表示可丢弃的外部代码。
原始修改时间用于迁入顺序和来源索引，Git 提交时间记录实际迁入时间。
源码、文档、规约、迁移工具与验收摘要分类提交。所有必要未跟踪文件
都应纳入对应分支；大权重、完整数据、缓存、原始日志和私有快照不进公开 Git。
