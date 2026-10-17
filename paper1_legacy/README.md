# omniultra

统一 Engine（配置 + state + callbacks）下的超声场景重建与实验仓库（UltraGS、FiMA 等）。

文档索引见 `docs/README.md`；脚本分类见 `scripts/README.md`。

## 环境与算力

集群上拿 GPU、激活环境与 CUDA 模块：见 **`docs/operations/slurm_skills.md`**（`srun` + `conda activate gaussian_splatting` + `module load cuda/11.8.0`）。

## 实验结论备忘（当前阶段）

以下结论来自本仓库内脚本与对话中的实测，便于后续设计取舍；**不是**对所有可能实现的普适定理。

### 1. Batch 化 CUDA 算子：有收益、能加速

- 历史调试入口已清理；需要复查时看 git history。当前可通过对应 Hydra experiment alias 和 override，在固定「等效图像步数」对齐的前提下，对比 **单帧 baseline**（`batched_kernel=false`, `batch_size=1`）与 **批处理路径**（`batched_kernel=true`, 如 `batch_size=4`）。
- 结论方向：**启用批处理 CUDA 路径可缩短墙钟时间**（具体倍率随 GPU、分辨率与配置而变，以你本地日志为准）。

### 2. Local window：在当前实现与指标下整体不划算

**UltraGS（高斯 / 渲染侧）**

- `model.offline.local_window` 路径涉及对高斯做窗口归属、子集与合并等逻辑，**额外 CPU/GPU 与工程开销**往往抵消「局部化」带来的理论收益；在现有 profiling 下表现为 **整体变慢**而非加速。
- 边界接缝 / joint 等扩展只会再加重负担；详见 **`docs/methods/baselines/ultrags_baseline/local_window_jointopt_ab_result.md`**（同配置下 joint 开启约 **24%** 墙钟变慢、画质差异不形成明确优势）。

**PEM / FiMA（相邻帧姿态侧）**

- 历史 FiMA phase-A 脚本已清理；当前通过 `python run.py experiment=preproduction/baseline_reproduction/fima/paper_edges` 及相关 override，在 holdout **160 train / 40 test**、相邻双图 PEM、同一链式评测协议下对比
  - **`phase_a_full`**：所有「全局相邻且两端均在 train」的边上训练，eval 上从第 0 帧链式乘到各 test 帧；
  - **`phase_a_local_window`**：长度 8、`stride=4` 的滑窗内采样训练，eval 时窗内前向 + 重叠边 6D 平均合并后再链式对齐。
- 一次代表性结果（`FIMA_ITERATIONS=2500`, `SEED=42`，链式指标 `eval/pem_chain/val`）：**full 优于 local-window**（例如平移 RMSE 约 **3.2 mm vs 5.4 mm**，旋转平均角约 **0.12 rad vs 0.20 rad**）。
- 含义：**「用局部窗训相邻位姿」这条叙事在当前协议下损害链式精度**；与 GS 侧的效率结论叠加后，**local window 在本仓库主线里不宜再作为默认卖点**。

### 3. PEM ↔ UltraGS 消融（B0 / B1）与 OCC 融合

**设定（便于复现时对齐量级）**：`synthetic_liver_x2`、`holdout_interval`（如 160 train / 40 test）、UltraGS **关闭离线密度控制（DC）**、初始化约 **50 万高斯**，PEM/GS 步数以当前 Hydra alias 和 override 为准；以下为该设定下、本仓库实测得到的**取向性结论**，非普适定理。

- **B0**：子进程训 PEM → 导出链式 `(N,4,4)` pose → 本进程 UltraGS 训练（**无 OCC 回灌**）。
- **B1**：``PemGsAblationB1AlternatingTrainCallback`` 驱动多轮 PEM↔GS **子进程**交替；上一轮 GS 导出 occupancy 写入 `model.pem.gs_occ_cache_dir` 供下一轮 PEM 使用（编排见 ``methods/ultrags/tmp_pem_gs_ablation_b0_b1_callbacks.py``）。

**两条关键取向**：

1. **交替 vs 一条龙（首轮对齐）**：在 **B1 第一轮 GS（尚未用上轮 OCC）** 与 **B0** 对比，holdout PSNR/SSIM **基本持平**——说明在给定实现里，**「PEM/GS 子进程轮换」本身**并不比「先把 PEM 训完再一次性喂 GS」在第一轮上就亏掉明显重建质量；（是否长期、多轮、不同总步数下仍成立需更多实验与末步评测收口。）

2. **OCC 进 PEM：`concat` vs `weighted`**：默认 **`model.pem.occ_fusion_mode=concat`** 为 **[灰度, OCC] 通道拼接**（2ch stem）。在闭环 **第二轮**，该方式相对首轮更易出现退化；改用 **`weighted`** ——灰度乘以 `clamp(1 + scale * O, lo, hi)` 再进 **单通道** stem ——在同一对比设置下 **第二轮指标优于 concat**。**若产品上要走 OCC 回灌 PEM，更应优先加权模式**；写正式论文/对外报告建议补 **多 seed** 以及在 **训练结束步**对最终 checkpoint **再跑一次 test**。（**单笔最优仍可能是 B0 式一刀流**——加权解决的是闭环里 concat 的常见坑，不自动保证全面超过 B0。）

入口与环境变量：历史 B0/B1 脚本已归档；新实验默认使用 `python run.py experiment=preproduction/baseline_reproduction/fima/paper_edges`，OCC 融合通过 Hydra override 调整。

### 4. 相关脚本与配置索引

| 主题 | 入口 |
|------|------|
| 批 CUDA A/B | 历史调试脚本已清理；按 git history 复查，当前新实验使用 Hydra alias + override |
| GS local window 调试矩阵 | 历史调试脚本已清理；当前新实验使用 Hydra alias + override |
| FiMA PEM 窗长消融 | 历史调试脚本已清理；FiMA 相关工具见 `scripts/methods/baselines/fima/` |
| FiMA 相邻帧阶段 A（full vs local-window 链式） | `python run.py experiment=preproduction/baseline_reproduction/fima/paper_edges` |
| PEM ↔ UltraGS B0/B1（OCC concat / weighted） | 历史 B0/B1 脚本已清理；当前实验入口使用 Hydra alias |

---

历史凭据备忘已从公开迁移副本移除；原件保存在本地迁移私有快照。
