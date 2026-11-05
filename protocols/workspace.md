# 本机工作区与资源规约

本机以 [/home/yibo/AGENTS.md](../../../AGENTS.md) 和 [paper1/AGENTS.md](../../AGENTS.md) 为权威规则。本文件取代旧工程文档中的 Python 3.7、Hydra-only、独立 protocol 仓库和本机必须使用 Slurm 的描述。

代码位于 nerfstudio 与 gsplat 两个 fork，当前开发分支为 `refactor/native-ultrasound`。docs 使用独立 Git 仓库，outputs 保存不跟踪的运行产物；独立 results 已取消。可执行配置和实验矩阵归入 Nerfstudio，不新设一级 protocol 目录。

所有入口通过 `../../bin/paper1` 选择 paper1 源码。共享 Python/Torch 在 shared/envs，项目覆盖层只路由源码。读取 registry.json 并现场核实工具链；安装变更持 provision.lock 独占锁，编译和运行持共享锁。GPU 使用前检查实际占用并取得对应设备锁，不伪造 Slurm 分配。

数据正文在 shared/datasets，模型在 shared/models，环境在 shared/envs，缓存在 shared/cache。R2 只保存正式发布数据和明确验收冻结的关键权重，禁止用来备份代码、文档、环境和一般输出。批量云操作先核查共享请求预算。

历史 checkpoints 不迁入；源端两目录保留，不实际删除。当前任务停在正式 1M/30k 之前，不能因迁移验收完成而自动启动训练。
