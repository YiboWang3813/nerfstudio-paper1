# Paper1 原生运行入口

从 `/home/yibo/paper1/bin/paper1` 运行。入口选择本项目 Nerfstudio/gsplat
源码、已有 CUDA 12.4 和共享 Torch 2.5.1；不会替换 paper2 的源码入口。
训练、编译和测试自动持有共享环境读锁并租用空闲物理 GPU。

```bash
./bin/paper1 identity
./bin/paper1 test -q
./bin/paper1 preflight --output migration/preformal-data-config-validation.json
./bin/paper1 prepare --method ours \
  --manifest /home/yibo/shared/datasets/synthetic_liver/p100k-native-v2/l1/manifest.json \
  --output /home/yibo/paper1/outputs/formal \
  --config /home/yibo/paper1/migration/plans/ours-liver-l1.yml
```

`prepare` 只生成官方 Nerfstudio 配置和检查回执，不启动训练。正式预算为
Gaussian 方法 1,000,000 个初始点、30,000 次更新；历史 `p100k` 协议名不代表
当前点数。生成的配置可由官方 Trainer 的 `--load-config` 加载。
Nerfstudio 的 step 从 0 开始，因此第 30,000 次更新的最终评估在 step=29,999
触发；不把旧引擎的一基计数直接当成原生评估间隔。
本次迁移验收到正式实验之前停止，不运行正式配置。

当前比较方法为 Ours、UltraGauss、UltraGray、GaussianPile、Slice-adapted
3DGS、UltraGS、Nerfus、UltraNeRF。模型使用原生 Model/Field，数据通过
DataParser/DataManager 提供完整切片；训练、调度、检查点由原生 Trainer
负责，密度控制通过 gsplat Strategy 和优化器状态操作实现。`paper1_legacy`
仅保留来源代码和配置参考；原生执行不导入其 Engine/Scene/方法注册表。
早期研究分支不冒充已验收的当前比较方法。

`prepare` 支持 Ours 的局部条件化、全局衰减、增密和剪枝消融选项，以及
`--pose-mode canonical`。canonical 默认使用冻结的 elevational spacing，
不从文件位姿估计间距。路径剪枝与关闭全局衰减的矛盾组合直接报错。
位姿残差、帧权重和 MC 边缘化配置由原生配置类提供，外部输入校验和保存在
检查点中。离线 `convert_protocols.py` 可以重新解析历史配置以核对原生快照；
Hydra 不参与原生训练。

数据正文在 `shared/datasets/<dataset>/p100k-native-v2/`。四类数据共
78 个 sweep；脊柱 trim=8，其余 trim=10，测试比例 0.2。保留 v1 及来源回执。
肝脏 l2/l3 的源变换含约 1% 非正交性，正式配置显式保留原始仿射几何，
不会投影成另一套旋转。发布工具只处理单个数据集版本，按包上传、读回
SHA256 校验，最后更新清单和 current；请求账本是保守预留，并非账户账单。
原始数据及已验证副本没有删除。

正式评估固定 pyiqa 0.1.7，报告 PSNR、L1、SSIM、LPIPS、DISTS、GMSD，
逐帧 CSV 附带原始 frame ID、数据版本、清单哈希和执行源码身份。
训练的 padded-window SSIM 与评估 SSIM 分别记录。公开指标网络仅存放于
`shared/models`，不会进入训练参数或实验检查点。评估隔离训练 RNG，
包括 UltraGray 在零扰动幅度下仍然发生的随机抽样。

新检查点记录采样器、RNG、密度控制和执行来源；拓扑变化后恢复参数及
优化器状态。不会加载旧实验权重。UltraGauss 强度初始化使用浮点开区间，
修复像素值 1 导致无限 logit 的问题。CPU 上按原分布选择初始化像素，避免
把整段 sweep 的全部三维坐标和属性同时放到 GPU；不要求新旧权重逐值相同。

`smoke_matrix.py` 只做 256 点、3 次更新和第 4 次续训验证，覆盖四类数据和
八个方法，属于工程检查。历史文稿、图表与科学结果验收仍待正式复现。
